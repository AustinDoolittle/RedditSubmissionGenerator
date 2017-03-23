import praw
import argparse as ap
import secrets
import sys
import os
import urllib
import cStringIO as csio
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import imutils
import cv2
import numpy as np
from PIL import Image

DEF_CLASS_FILE = "Keys/subreddits.keys"
PKL_CACHE_DIR = "Cache/"

__verbose = False

def load_subreddits(filename):
  try:
    with open(filename, 'r') as f:
      return [r.strip() for r in f.readlines()]
  except:
    return []


def get_submission_data(subreddits, reddit):
  global __verbose

  retval = {}
  for subreddit in subreddits:
    if __verbose:
      print "\tGetting submissions for r/" + subreddit

    count = 1
    retval[subreddit] = []
    for submission in reddit.subreddit(subreddit).top(limit=1000):
      if __verbose:
        print "\t\tSubmission " + str(count)
      count += 1

      retval[subreddit].append({
          'id': submission.fullname,
          'link': submission.url,
          'title': submission.title
        })

    if __verbose:
      print "\n"

  return retval

#retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
def image_to_feature_vector(image, size=(64, 64)):
  # resize the image to a fixed size, then flatten the image into
  # a list of raw pixel intensities
  return cv2.resize(image, size).flatten()

#retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
def extract_color_histogram(image, bins=(8, 8, 8)):
  # extract a 3D color histogram from the HSV color space using
  # the supplied number of `bins` per channel
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
    [0, 180, 0, 256, 0, 256])
 
  # handle normalizing the histogram if we are using OpenCV 2.4.X
  if imutils.is_cv2():
    hist = cv2.normalize(hist)
 
  # otherwise, perform "in place" normalization in OpenCV 3 (I
  # personally hate the way this is done
  else:
    cv2.normalize(hist, hist)
 
  # return the flattened histogram as the feature vector
  return hist.flatten()

def main(argv):
  global __verbose

  #parse arguments
  parser = ap.ArgumentParser(description="This application retrieves submissions from Reddit using the Reddit API")
  parser.add_argument("--classes", default=DEF_CLASS_FILE, help="The file to load the list of classes from (default = 'Keys/subreddits.keys')")
  parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Increase verbosity of the application")
  
  args = parser.parse_args(argv)

  __verbose = args.verbose

  print "Creating Reddit object"

  #setup reddit object
  reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                      client_secret=secrets.CLIENT_SECRET,
                      password=secrets.CLIENT_PASSWORD,
                      user_agent=secrets.CLIENT_USER_AGENT,
                      username=secrets.CLIENT_USERNAME)

  print "Loading subreddits from file " + args.classes

  #Load the Subreddit Keys
  subreddits = load_subreddits(args.classes)

  if subreddits == []:
    print "There was an error reading the classes file"
    sys.exit(-2)

  print "Retrieving submission data from subreddits...\n"

  raw_class_filename = os.path.basename(args.classes)

  save_to_cache = False

  try:
    with open(PKL_CACHE_DIR + "/" + raw_class_filename + ".pkl", 'rb') as f:
      submissions = pkl.load(f)
      print "\tRetrieved from Cache\n"
  except:
    print "\tNo cached data, retrieving from Reddit API\n"
    submissions = get_submission_data(subreddits, reddit)
    save_to_cache = True

  if save_to_cache:
    try:
      with open(PKL_CACHE_DIR + "/" + raw_class_filename + ".pkl", 'wb') as f:
        pkl.dump(submissions, f, pkl.HIGHEST_PROTOCOL)
    except:
      print "Unable to save submissions to cache, oh well..."

  for subreddit in submissions:
    for submission in submissions[subreddit]:
      file = csio.StringIO(urllib.urlopen(submission['link']).read())
      img = Image.open(file)
      img.show()
      raw_input()

  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files
  sys.exit(main(sys.argv[1:]))


