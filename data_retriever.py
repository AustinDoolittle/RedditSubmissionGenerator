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
import imutils
import cv2
import numpy as np
from PIL import Image

DEF_CLASS_FILE = "Keys/subreddits.keys"
PKL_CACHE_DIR = "Cache/"
DEF_NEIGHBORS = 1
DEF_JOBS = -1
DEF_SUB_COUNT = 1000

__verbose = False

def load_subreddits(filename):
  try:
    with open(filename, 'r') as f:
      return [r.strip() for r in f.readlines()]
  except:
    return []


def get_submission_data(subreddits, reddit, sub_count):
  global __verbose

  retval = {}
  for subreddit in subreddits:
    if __verbose:
      print "\tGetting submissions for r/" + subreddit

    count = 1
    retval[subreddit] = []

    for submission in reddit.subreddit(subreddit).top(limit=sub_count):
      if __verbose:
        print "\t\tr/" + subreddit + ": " + str(count) + "/" + str(sub_count)
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
def image_to_feature_vector(image, size=(32, 32)):
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

def process_images(submissions):
  global __verbose

  raw_imgs = []
  features = []
  classes = []


  for subreddit in submissions:
    counter = 1
    total_count = len(submissions[subreddit])
    if __verbose:
      print "\tProcessing images from r/" + subreddit

    for submission in submissions[subreddit]:

      #Code augmented from http://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format, user: Abhishek Thakur
      try:
        file = csio.StringIO(urllib.urlopen(submission['link']).read())
        pil_img = Image.open(file)
      except IOError:
        print "\t\t~Error Retrieving image, skipping..."
        total_count -= 1
        continue

      cv_img = np.array(pil_img)

      #convert from RGB to BGR
      try:
        cv_img = cv_img[:, :, ::-1].copy() 
      except IndexError:
        print "\t\t~Error Converting from RGB to BGR, skipping..."
        total_count -= 1
        continue

      if len(cv_img.shape) < 3:
        print "\t\t~Image does not have enough channels, skipping..."
        total_count -= 1
        continue

      pixels = image_to_feature_vector(cv_img)
      hist = extract_color_histogram(cv_img)

      if hist is None:
        print "\t\t~Error Converting raw image to histogram, skipping..."
        total_count -= 1
        continue

      raw_imgs.append(pixels)
      features.append(hist)
      classes.append(subreddit)

      if __verbose:
        print "\t\tr/" + subreddit + ": " + str(counter) + "/" + str(total_count)
      counter += 1

    return raw_imgs, features, classes

def train_and_test(train_set, train_labels, test_set, test_labels, neighbor_count, job_count):
  classifier = KNeighborsClassifier(n_neighbors=neighbor_count, n_jobs=job_count)
  classifier.fit(train_set, train_labels)
  return classifier.score(test_set, test_labels)

def main(argv):
  global __verbose

  #parse arguments
  parser = ap.ArgumentParser(description="This application retrieves submissions from Reddit using the Reddit API")
  parser.add_argument("--classes", default=DEF_CLASS_FILE, help="The file to load the list of classes from (Default = '" + DEF_CLASS_FILE + "'')")
  parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Increase verbosity of the application")
  parser.add_argument("--neighbors", default=DEF_NEIGHBORS, help="The number of neighbors to observe when classifying (Default = '" + str(DEF_NEIGHBORS) + "')")
  parser.add_argument("--jobs", default=DEF_JOBS, help="The number of cores to run the classification processes (Default = " + str(DEF_JOBS) + ")")
  parser.add_argument("--subcount", default=DEF_SUB_COUNT, help="The amount of submissions from each subreddit to pull (Default = " + str(DEF_SUB_COUNT) + ") **BUG IN LIBRARY, OPEN ISSUE: https://github.com/praw-dev/praw/issues/759**")
  
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

  raw_class_filename = os.path.splitext(os.path.basename(args.classes))[0]
  pkl_file = PKL_CACHE_DIR + "/" + raw_class_filename + "_" + str(args.subcount) + ".pkl"

  save_to_cache = False

  try:
    with open(pkl_file, 'rb') as f:
      submissions = pkl.load(f)
      print "\tRetrieved from Cache\n"
  except Exception as ex:
    print "\tNo cached data, retrieving from Reddit API \n"
    submissions = get_submission_data(subreddits, reddit, args.subcount)
    save_to_cache = True

  if save_to_cache:
    print "Saving Cache to " + pkl_file
    try:
      with open(pkl_file, 'wb') as f:
        pkl.dump(submissions, f, pkl.HIGHEST_PROTOCOL)
    except Exception as ex:
      print "\t~Unable to save submissions to cache: " + ex.strerror

  print "Processing images..."
  raw_imgs, features, classes = process_images(submissions)
  print "Image Processing Complete"


  print "Shuffling and creating datasets"
  #create test and train sets
  train_imgs, test_imgs, train_img_labels, test_img_labels = train_test_split(raw_imgs, classes, test_size=0.25, random_state=42)
  train_feats, test_feats, train_feat_labels, test_feat_labels = train_test_split(features, classes, test_size=0.25, random_state=42)

  #Test classification accuracy
  raw_img_acc = train_and_test(train_imgs, train_img_labels, test_imgs, test_img_labels, args.neighbors, args.jobs)
  feat_acc = train_and_test(train_feats, train_feat_labels, test_feats, test_feat_labels, args.neighbors, args.jobs)

  print "~~FINISHED~~"
  print "Raw Image Accuracy: " + str(raw_img_acc * 100) + "%"
  print "Feature Accuracy: " + str(feat_acc * 100) + "%"

  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files
  sys.exit(main(sys.argv[1:]))


