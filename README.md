# RedditSubmissionGenerator
This application uses a convolutional neural network and multiple recursive neural networks to generate posts to reddit to fitting subreddits based on the content of their images

## Dependencies
- pickle
- cv2
- urllib
- numpy
- argparse
- praw

## Reddit API
You must set up an application through the reddit API and include the following information as variables in a file in the root directory called secrets.py
- Client ID => CLIENT_ID
- Client Secret => CLIENT_SECRET
- Password => CLIENT_PASSWORD
- User Agent => CLIENT_USER_AGENT
- Username => CLIENT_USERNAME
