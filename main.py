#!/usr/bin/env python
import argparse as ap
import sys
import os
import subprocess
from trainer import Trainer

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
                        'u/IncrediibleHulk',
                        )

def parse_args(argv):
    # parse arguments
    parser = ap.ArgumentParser(description='This application retrieves submissions from Reddit using the Reddit API \
                                          and trains a neural network to classify the images by subreddit')
    parser.add_argument('--overwrite-cache', default=False, action='store_true',
        help='Presence of this flag ignores the cache and pulls data from reddit')
    parser.add_argument('--dataset-dir', default='./datasets',
        help='The directory where tfrecord files should be saved')
    parser.add_argument('--subreddits-file', required=True,
        help='The file containing the subreddits to pull from and classify')
    parser.add_argument('--dataset-tag', default='default',
        help='The tag used to differentiate between different tfrecord configurations')
    parser.add_argument('--submission-count', default=100, type=int,
        help='The amount of post from each subreddit to retrieve')
    parser.add_argument('--num-epochs', default=10, type=int,
        help='The number of epochs to run during training')
    parser.add_argument('--new-size', default=(256, 256), nargs=2, type=int,
        help='The size of the new image to resize')
    parser.add_argument('--train', action='store_true', default=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    trainer = Trainer(args)
    trainer.train()



if __name__ == '__main__':
    # call the main method, do this so we can also call the main method from other files if need be
    main(sys.argv[1:])
