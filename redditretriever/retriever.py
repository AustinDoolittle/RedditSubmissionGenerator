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

    raw_class_filename = os.path.splitext(os.path.basename(self.filename))[0]
    pkl_file = config.PKL_CACHE_DIR + raw_class_filename + "_" + str(sub_count) + ".pkl"
    save_to_cache = False

    #attempt to open the file
    try:
      with open(pkl_file, 'rb') as f:
        submissions = pkl.load(f)
        print "\tRetrieved from Cache\n"
    except Exception as ex:
      #the file did not exist or could not be opened, we'll get data from Reddit
      print "\tNo cached data, retrieving from Reddit API \n"
      submissions = self.get_submission_data(sub_count)
      save_to_cache = True

    #check if we should cache the data we just got
    if save_to_cache:
      print "Saving Cache to " + pkl_file
      try:
        with open(pkl_file, 'wb') as f:
          pkl.dump(submissions, f, pkl.HIGHEST_PROTOCOL)
      except Exception as ex:
        print "\t~Unable to save submissions to cache: " + str(ex)

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




