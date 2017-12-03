import tensorflow as tf
import praw
import os
import sys
import subprocess
from io import BytesIO, StringIO
import random
import re
import requests
from PIL import Image
import pickle
from collections import namedtuple


oauth_creds_keys = [
    'client_id',
    'client_secret',
    'client_password',
    'user_agent',
    'username',
]

RedditOAuthCredentials = namedtuple('RedditOAuthCredentials', oauth_creds_keys)

EOT = 'EOT'

# declare helper methods for converting to TFRecords and back
# NOTE: these features aren't documented by tensorflow, good luck
def _bytes_feature(value):
    compat_value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[compat_value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    return tf.train.FeatureList(feature=[_int64_feature(x) for x in value])

def add_submission_to_tfrecord(writer, image_url, encoded_title, new_size, label, labels):
    """Serialize image and encoded title to TFRecord

    This method will also encode the image to png and scale it to *new_size*

    *writer* -- A TFRecordWriter instance
    *image_url* -- The url of the image to convert. If this url does not yield a valid image,
        a ValueError is thrown (i think)
    *encoded_title* -- The title of the post encoded to a list of the index of each word in the
        vocabulary of the total lexicon
    *new_size* -- the new size to scale the image to


    """
    def get_scaled_image_from_url(image_url):
        response = requests.get(image_url)
        raw_image_file = BytesIO(response.content)
        raw_image = Image.open(raw_image_file).convert('RGB')
        scaled_image = raw_image.resize(new_size, Image.ANTIALIAS)
        scaled_image_str = BytesIO()
        scaled_image.save(scaled_image_str, format='png')
        return scaled_image_str.getvalue()

    image_str = get_scaled_image_from_url(image_url)
    sorted_labels = sorted(labels)
    context = tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_str),
        'image/url': _bytes_feature(image_str),
        'label/text': _bytes_feature(label),
        'label/index': _int64_feature(sorted_labels.index(label))
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        'title': _int64_feature_list(encoded_title),
    })

    # We use SequenceExample because the titles are varying in size
    example = tf.train.SequenceExample(
        context=context,
        feature_lists=feature_lists
    )
    writer.write(example.SerializeToString())


def parse_fn(example):
    "Defines the parse function to be mapped to the output of the dataset node"
    context_features = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
        'image/url': tf.FixedLenFeature([], dtype=tf.string),
        'label/text': tf.FixedLenFeature([], dtype=tf.string),
        'label/index': tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        'title': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    context_parsed.update(sequence_parsed)
    return context_parsed

class LockedListException(Exception):
    pass

class LockableList(list):
    "A list that allows you to lock it so that no more values can be added"
    def __init__(self, *args, **kwargs):
        self.locked = False
        super(LockableList, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if self.locked:
            raise LockedListException('key ' + key + ' can not be set because the list is currently locked')
        else:
            super(LockableList, self).__setitem__(key, value)

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

class SubmissionDataset(object):
    """Class that defines a on-disk dataset

    This class is useful for retrieving and storing images from a collection of subreddits
    without needing to download the entire set of images. If no TFRecord is created at the
    path derived from *ds_root* and *tag*, a new one is created. Otherwise, the cached version
    is used.

    This class also provides a dataset node via *get_dataset_node* that can be used to create
    a reinitializable iterator that is useful for training.
    """
    def __init__(self, subreddits, oauth_creds, conv_opts=None, submission_count=100,
                ds_root='./datasets', tag='default', random_seed=None):
        """SubmissionDataset init

        *subreddits* -- a list of subreddits to pull submissions from
        *oauth_creds* -- a RedditOAuthCredentials namedtuple of Reddit's OAuth credentials
        *conv_opts* -- a dictionary containing custom conversion options
        *submission_count* -- the amount of submissions to attempt to pull from each subreddit.
            it is important to note that not all submissions will be successfully recovered, especially
            from subreddits that are heavy in non-image links
        *ds_root* -- the root directory that the dataset may/will be saved
        *tag* -- a tag that is added to the .tfrecord file to differentiate between different
            configurations
        *random_seed* -- the random seed to use when accessing psuedorandom functions

        """
        self.subreddits = LockableList(subreddits)
        self.subreddits.lock()

        self.reddit = praw.Reddit(**oauth_creds._asdict())
        self.ds_root = ds_root
        self.tag = tag
        self.submission_count = submission_count
        conv_opts = conv_opts or {}
        self.metadata_attrs = ['conv_opts', 'vocab', 'subreddits']
        self.conv_opts = {
            'new_size': (256, 256),
            'train_test_split': [0.8, 0.2],
        }

        self.dataset_nodes = {}

        # validate conv_opts
        self._validate_conv_opts()

        # use not None in case the random seed is 0
        if random_seed is not None:
            random.seed(random_seed)

        self.try_load_metadata()

    def _validate_conv_opts(self):
        #TODO maybe create helper class to make this more graceful
        if len(self.conv_opts['new_size']) != 2:
            raise ValueError('conv_opts key new_size must be of length 2')
        if not all(x > 0 for x in self.conv_opts['new_size']):
            raise ValueError('conv_opts key new_size must be two values that are > 0')
        elif self.conv_opts['new_size'][1] <= 0:
            raise ValueError('conv_opts key resized_width must be > 0')
        elif len(self.conv_opts['train_test_split']) != 2:
            raise ValueError('conv_opts key train_test_split must be a tuple/list of length 2')
        elif sum(self.conv_opts['train_test_split']) != 1:
            # WARN possible floating point arithmetic error?
            raise ValueError('conv_opts key train_test_split contents must have a sum of 1')

    def try_purge_cache(self):
        for ds_type in ['train', 'test']:
            try:
                os.delete(self.get_tfrecord_path(ds_type))
            except OSError:
                pass

        try:
            os.delete(self.get_metadata_path())
        except OSError:
            pass

    def get_metadata_path(self):
        return '{}_meta.pkl'.format(self.tag)

    def save_metadata(self):
        """Save config settings of this Submissionataset instance.

        The properties of the current instance will be stored in a pickle-file housed in the folder
        of the dataset which this instance represents.

        """
        metadata_path = self.get_metadata_path()
        # build pickle dictionary
        pickle_dict = {}
        for attr in self.metadata_attrs:
            pickle_dict[attr] = getattr(self, attr)

        # write dict to metadata file
        with open(metadata_path, 'wb') as f:
            pickle.dump(pickle_dict, f)

    def try_load_metadata(self):
        "Load config settings of this RawDataset instance"
        metadata_path = self.get_metadata_path()
        try:
            # load pickle dictionary
            with open(metadata_path, 'rb') as f:
                pickle_dict = pickle.load(f)

            # unpack dict into attributes of self
            self.__dict__.update(pickle_dict)
        except (IOError, EOFError):
            pass

    def get_tfrecord_path(self, dataset_type):
        "Returns the tf_record path for the given dataset_type"
        assert(dataset_type in ['train', 'val', 'test'])
        tfrecord_name = '{}-{}.tfrecord'.format(self.tag, dataset_type)
        return os.path.join(self.ds_root, tfrecord_name)

    def split_submissions(self, submission_dict):
        """Splits all submissions into training and testing set

        Note that this also ensures an equal sampling of each subreddit is in the test and train set

        Returns two lists of submissions in the following format:
            [
                {
                    'title': 'me irl',
                    'url': 'i.reddit.com/image',
                    'subreddit': 'me_irl'
                },
                {
                    'title': 'I took this picture',
                    'url': 'imgur.com/image',
                    'subreddit': 'pics'
                },
                ...
            ]
        """
        train_ds = []
        test_ds = []
        for subreddit in submission_dict:
            # TODO kinda hacky
            sub_list = submission_dict[subreddit]
            for submission in sub_list:
                submission.update({'subreddit': subreddit})

            random.shuffle(sub_list)
            split_index = int(len(sub_list) * self.conv_opts['train_test_split'][0])
            train_ds += sub_list[:split_index]
            test_ds += sub_list[split_index:]

        return train_ds, test_ds

    def get_submissions(self):
        """Get submissions from the subreddits in *self.subreddits*

        The return format of this function is a dictionary with an entry for each subreddit.
        The value of the subreddits' entries is a list of objects containing the url and title
        of the post. For Example:
            {
                'awww': [
                    {
                        'title': 'Cute Dog Picture',
                        'url': 'imgur.com/cutedogpicture'
                    },
                    {
                        'title': 'Cute Cat Picture',
                        'url': 'imgur.com/cutecatpicture'
                    },
                    ...
                ],
                ...
            }

        """
        submission_dict = {}
        vocab = []
        for subreddit_name in self.subreddits:
            submission_dict[subreddit_name] = []
            counter = 1
            subreddit = self.reddit.subreddit(subreddit_name)

            for submission in subreddit.top(limit=self.submission_count):
                sub_info = {
                            'title': submission.title,
                            'url': submission.url,
                }
                submission_dict[subreddit_name].append(sub_info)

                # might as well create our vocab while we're at it
                vocab += self.parse_title_string(sub_info['title'])
        vocab = list(set(vocab))

        # NOTE: assumes that the word 'EOT' doesn't appear in any of the titles
        vocab.append(EOT) # end of title delimiter
        return submission_dict, set(vocab)

    def parse_title_string(self, title):
        "Seperates out all of the ascii words and punctuation into a list"
        # TODO allow for word wise or character wise generation
        all_matches = re.findall(r"[\w']+|[+.,!?;-]", title)
        if not isinstance(all_matches, list):
            all_matches = [all_matches]
        return all_matches

    def encode_to_vocab(self, title_string):
        "Encodes a string to a vocab index array"
        parsed_title = self.parse_title_string(title_string)
        encoded_title = []
        for item in parsed_title:
            try:
                encoded_title.append(self.vocab.index(item))
            except ValueError:
                # don't encode a word if it isn't in the vocab
                pass

        # end it with an EOT
        encoded_title.append(self.vocab.index(EOT))
        return encoded_title

    def maybe_create_tfrecords(self, force=False):
        """This possible creates a tfrecord file

        If the train tfrecord path exists and *force* is false, then the tfrecord is assumed to
        have been created. If that tfrecord path does not exist or force is true, then the old
        tfrecord is purged and a new tfrecord is birthed from its ashes.
        """
        reusing = not force and os.path.isfile(self.get_tfrecord_path('train'))

        if not reusing:
            #load tfrecord
            submission_dict, vocab = self.get_submissions()

            self.vocab = LockableList(vocab)
            self.vocab.lock()

            train_ds, test_ds = self.split_submissions(submission_dict)
            for ds_name, ds_list in [('train', train_ds), ('test', test_ds)]:
                print('Converting', ds_name)
                if not os.path.isdir(self.ds_root):
                    os.makedirs(self.ds_root)

                tfrecord_path = self.get_tfrecord_path(ds_name)
                counter = 0
                total = len(ds_list)
                with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
                    for data_instance in ds_list:
                        encoded_title = self.encode_to_vocab(data_instance['title'])

                        try:
                            add_submission_to_tfrecord(
                                writer,
                                data_instance['url'],
                                encoded_title,
                                self.conv_opts['new_size'],
                                data_instance['subreddit'],
                                self.subreddits
                            )
                        except OSError:
                            total -= 1
                            continue

                        counter += 1
                        # TODO: sloppy progress printing, fix
                        print('\t', counter, '/', total)
            self.save_metadata()

        return not reusing

    def get_dataset_node(self, ds_type):
        "Returns a TFRecordDataset created from the tfrecord file that we have on tap"
        assert(ds_type in ['train', 'test'])

        if ds_type not in self.dataset_nodes:
            self.maybe_create_tfrecords()
            ds = tf.data.TFRecordDataset([self.get_tfrecord_path(ds_type)])
            ds = ds.map(parse_fn)
            self.dataset_nodes[ds_type] = ds

        return ds




if __name__ == '__main__':
    client_id = os.environ['REDDIT_CLIENT_ID']
    client_secret = os.environ['REDDIT_CLIENT_SECRET']
    client_password = os.environ['REDDIT_CLIENT_PASSWORD']
    user_agent = 'test_user_agent'
    username = os.environ['REDDIT_CLIENT_USERNAME']

    oauth_creds = RedditOAuthCredentials(
        client_id,
        client_secret,
        client_password,
        user_agent,
        username
    )

    ds = SubmissionDataset(['awww'], oauth_creds)
    ds_node = ds.get_dataset_node('train')
    ds_node = ds_node.shuffle(buffer_size=10000)
    ds_node = ds_node.batch(1)
    iterator = ds_node.make_one_shot_iterator()

    with tf.Session() as sess:
        result = sess.run(iterator.get_next())
        print(result)
