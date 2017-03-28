import secrets
import praw
import os
import config
import pickle as pkl

class SubmissionRetriever(object):
  def __init__(self, client_id, client_secret, password, user_agent, username, filename=None, verbose=False):
    self.reddit = praw.Reddit(client_id=client_id,
                      client_secret=client_secret,
                      password=password,
                      user_agent=user_agent,
                      username=username)
    if filename is not None:
      self.filename = filename
      self.load_subreddits(filename)

    self.verbose = verbose

  def load_subreddits(self, filename):
    try:
      with open(config.ROOT + filename, 'r') as f:
        #return lines with leading and trailing empty chars removed
        temp = [r.strip() for r in f.readlines()]
    except IOError:
      #error, return an empty list
      raise ValueError("The file " + filename + " does not exist")

    if len(temp) == 0:
      raise ValueError("The file " + filename + " is empty")

    self.subreddits = temp
    self.class_file = filename

  def get_submissions(self, sub_count):
    if self.subreddits is None:
      raise AttributeError("You must load the subreddits to get the submissions")

    

    submissions = self.get_submission_data(sub_count)

    return submissions

  def get_submission_data(self, sub_count):
    retval = {}

    #iterate over the subreddit list
    for subreddit in self.subreddits:
      if self.verbose:
        print "\tGetting submissions for r/" + subreddit

      count = 1
      retval[subreddit] = []

      #iterate over the submissions and gather their id, link, and title
      for submission in self.reddit.subreddit(subreddit).top(limit=sub_count):
        if self.verbose:
          print "\t\tr/" + subreddit + ": " + str(count) + "/" + str(sub_count)
        count += 1

        retval[subreddit].append({
            'id': submission.fullname,
            'link': submission.url,
            'title': submission.title
          })

      if self.verbose:
        print "\n"

    return retval 




