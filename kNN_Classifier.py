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
DEF_TEMP_IMG = "temp."
DEF_NEIGHBORS = 1
DEF_JOBS = -1
DEF_SUB_COUNT = 1000

__verbose = False

#Loads the subreddit names from the specified .keys file
def load_subreddits(filename):
  try:
    with open(filename, 'r') as f:
      #return lines with leading and trailing empty chars removed
      return [r.strip() for r in f.readlines()]
  except:
    #error, return an empty list
    return []

#Loops over the subreddits and gets the top sub_count submissions of all time
def get_submission_data(subreddits, reddit, sub_count):
  global __verbose
  retval = {}

  #iterate over the subreddit list
  for subreddit in subreddits:
    if __verbose:
      print "\tGetting submissions for r/" + subreddit

    count = 1
    retval[subreddit] = []

    #iterate over the submissions and gather their id, link, and title
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
 
  #normalize
  cv2.normalize(hist, hist)
 
  # return the flattened histogram as the feature vector
  return hist.flatten()

def process_images(submissions):
  global __verbose

  raw_imgs = []
  features = []
  classes = []

  #iterate over subreddits
  for subreddit in submissions:
    counter = 1
    total_count = len(submissions[subreddit])
    if __verbose:
      print "\tProcessing images from r/" + subreddit

    #iterate over submissions in subreddit
    for submission in submissions[subreddit]:

      try:
        #Get image
        cv_img = url_to_image(submission['link'])
      except (IOError, cv2.error):
        print "\t\t~Error Retrieving image, skipping..."
        total_count -= 1
        continue

      #check that the image was read successfully
      if cv_img is None:
        print "\t\tError Converting Image to CV2 format, skipping..."
        total_count -= 1
        continue

      #convert from RGB to BGR
      try:
        cv_img = cv_img[:, :, ::-1].copy() 
      except IndexError:
        #cv_img is structured incorrectly
        print "\t\t~Error Converting from RGB to BGR, skipping..."
        total_count -= 1
        continue
        
      #check that the image is not grayscale
      if len(cv_img.shape) < 3:
        print "\t\t~Image must have exactly 3 or 4 channels..."
        total_count -= 1
        continue

      #gather pixel and color histogram
      pixels = image_to_feature_vector(cv_img)
      hist = extract_color_histogram(cv_img)

      #check for histogram error
      if hist is None:
        print "\t\t~Error Converting raw image to histogram, skipping..."
        total_count -= 1
        continue

      #append to return values
      raw_imgs.append(pixels)
      features.append(hist)
      classes.append(subreddit)

      if __verbose:
        print "\t\tr/" + subreddit + ": " + str(counter) + "/" + str(total_count)
      counter += 1

  return raw_imgs, features, classes

#trains and tests on the provided train and test sets
def train_and_test(train_set, train_labels, test_set, test_labels, neighbor_count, job_count):
  classifier = KNeighborsClassifier(n_neighbors=neighbor_count, n_jobs=job_count)
  classifier.fit(train_set, train_labels)
  return classifier.score(test_set, test_labels)

def url_to_image(url):
  resp = urllib.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  return cv2.imdecode(image, cv2.IMREAD_COLOR)

def main(argv):
  global __verbose


  #parse arguments
  parser = ap.ArgumentParser(description="This application retrieves submissions from Reddit using the Reddit API")
  parser.add_argument("--classes", default=DEF_CLASS_FILE, help="The file to load the list of classes from (Default = '" + DEF_CLASS_FILE + "'')")
  parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Increase verbosity of the application")
  parser.add_argument("--neighbors", default=DEF_NEIGHBORS, type=int, help="The number of neighbors to observe when classifying (Default = '" + str(DEF_NEIGHBORS) + "')")
  parser.add_argument("--jobs", default=DEF_JOBS, type=int, help="The number of cores to run the classification processes (Default = " + str(DEF_JOBS) + ")")
  parser.add_argument("--subcount", default=DEF_SUB_COUNT, type=int, help="The amount of submissions from each subreddit to pull (Default = " + str(DEF_SUB_COUNT) + ")")
  args = parser.parse_args(argv)

  __verbose = args.verbose

  print "Creating Reddit object"

  #setup reddit object
  reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                      client_secret=secrets.CLIENT_SECRET,
                      password=secrets.CLIENT_PASSWORD,
                      user_agent=secrets.CLIENT_USER_AGENT,
                      username=secrets.CLIENT_USERNAME)

  #Load the Subreddit Keys
  print "Loading subreddits from file " + args.classes
  subreddits = load_subreddits(args.classes)

  #Check for error
  if subreddits == []:
    print "There was an error reading the classes file"
    sys.exit(-2)

  #Create pickle filename
  print "Retrieving submission data from subreddits...\n"
  raw_class_filename = os.path.splitext(os.path.basename(args.classes))[0]
  pkl_file = PKL_CACHE_DIR + raw_class_filename + "_" + str(args.subcount) + ".pkl"
  save_to_cache = False

  #attempt to open the file
  try:
    with open(pkl_file, 'rb') as f:
      submissions = pkl.load(f)
      print "\tRetrieved from Cache\n"
  except Exception as ex:
    #the file did not exist or could not be opened, we'll get data from Reddit
    print "\tNo cached data, retrieving from Reddit API \n"
    submissions = get_submission_data(subreddits, reddit, args.subcount)
    save_to_cache = True

  #check if we should cache the data we just got
  if save_to_cache:
    print "Saving Cache to " + pkl_file
    try:
      with open(pkl_file, 'wb') as f:
        pkl.dump(submissions, f, pkl.HIGHEST_PROTOCOL)
    except Exception as ex:
      print "\t~Unable to save submissions to cache: " + ex.strerror

  #Process all of the images
  print "Processing images..."
  raw_imgs, features, classes = process_images(submissions)
  print "Image Processing Complete"

  #Get Training and Test sets
  print "Shuffling and creating histogram training and test set"
  train_feats, test_feats, train_feat_labels, test_feat_labels = train_test_split(features, classes, test_size=0.25, random_state=42)

  print "Shuffling and creating raw image training and test set"
  train_imgs, test_imgs, train_img_labels, test_img_labels = train_test_split(raw_imgs, classes, test_size=0.25, random_state=42)

  #Test classification accuracy
  print "Training and testing histogram data"
  feat_acc = train_and_test(train_feats, train_feat_labels, test_feats, test_feat_labels, args.neighbors, args.jobs)

  print "Training and testing raw image data"
  raw_img_acc = train_and_test(train_imgs, train_img_labels, test_imgs, test_img_labels, args.neighbors, args.jobs)

  #Display results
  print "~~FINISHED~~"
  print "Raw Image Accuracy: " + str(raw_img_acc * 100) + "%"
  print "Feature Accuracy: " + str(feat_acc * 100) + "%"

  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files if need be
  sys.exit(main(sys.argv[1:]))


