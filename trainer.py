import tensorflow as tf
from dataset_util import SubmissionDataset, RedditOAuthCredentials
import os
import subprocess

def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def user_agent():
    return '{}_{}_{}'.format(
                        'reddit_submission_generator',
                        git_version(),
                        'IncrediibleHulk',
                        )

__client_id = os.environ['REDDIT_CLIENT_ID']
__client_secret = os.environ['REDDIT_CLIENT_SECRET']
__client_password = os.environ['REDDIT_CLIENT_PASSWORD']
__username = os.environ['REDDIT_CLIENT_USERNAME']

oauth_creds = RedditOAuthCredentials(
    __client_id,
    __client_secret,
    __client_password,
    user_agent(),
    __username
)
class Trainer(object):
    def __init__(self, args):
        self.args = args

    def _create_dataset(self):
        subreddits_file_no_ext = os.path.splitext(os.path.basename(self.args.subreddits_file))[0]
        ds_root = os.path.join(self.args.dataset_dir, subreddits_file_no_ext)

        try:
            with open(self.args.subreddits_file, 'r') as f:
                subreddits = f.read().splitlines()
        except OSError:
            print('ERROR could not open subreddits file', self.args.subreddits_file)
            raise

        conv_opts = {
            'new-size': self.args.new_size
        }
        self.ds = SubmissionDataset(
            subreddits,
            oauth_creds,
            conv_opts=conv_opts,
            ds_root=ds_root,
            submission_count=self.args.submission_count,
            tag=self.args.dataset_tag)

    def _get_data_instance(self, sess, iterator):
        instance_dict = sess.run(iterator)
        label = instance_dict['label/index']
        image = instance_dict['image/encoded']
        title = instance_dict['title']

        label_one_hot = tf.one_hot(label, len(self.ds.subreddits))
        title_one_hot = tf.one_hot(title, len(self.ds.vocab))
        image_decoded = tf.image.decode_png(image)

        return image_decoded, label_one_hot, title_one_hot

    def train(self):
        self._create_dataset()

        # create graph
        #TODO

        train_ds = self.ds.get_dataset_node('train')
        train_ds = train_ds.shuffle(buffer_size=10000)
        train_iter = train_ds.make_initializable_iterator()
        train_iter_next = train_iter.get_next()

        # start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.args.num_epochs):
                still_running = True
                sess.run(train_iter.initializer)
                while still_running:
                    try:
                        image, label, title = self._get_data_instance(sess, train_iter_next)
                        print(label)
                    except tf.errors.OutOfRangeError:
                        still_running = False

                # run a training interval

