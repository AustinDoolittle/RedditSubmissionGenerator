import numpy as np
import urllib
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum
import cv2

class ProcessType(Enum):
  FeatureVector = 0
  ColorHistogram = 1

class kNNClassifier(object):
  def __init__(self, verbose=False):
    self.verbose = verbose

  def url_to_image(self, url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

  def convert_to_dataset(self, submissions, type=ProcessType.ColorHistogram):
    retval = []
    classes = []

    #iterate over subreddits
    for subreddit in submissions:
      counter = 1
      total_count = len(submissions[subreddit])
      if self.verbose:
        print "\tProcessing images from r/" + subreddit

      #iterate over submissions in subreddit
      for submission in submissions[subreddit]:

        try:
          #Get image
          cv_img = self.url_to_image(submission['link'])
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

        #gather pixel or color histogram
        if type == ProcessType.FeatureVector:
          dat = self.image_to_feature_vector(cv_img)
        elif type == ProcessType.ColorHistogram:
          dat = self.extract_color_histogram(cv_img)
        else:
          raise ValueError("Process Type is invalid")

        #check for histogram error
        if dat is None:
          print "\t\t~Error Converting raw image to histogram, skipping..."
          total_count -= 1
          continue

        #append to return values
        retval.append(dat)
        classes.append(subreddit)

        if self.verbose:
          print "\t\tr/" + subreddit + ": " + str(counter) + "/" + str(total_count)
        counter += 1

    return train_test_split(retval, classes, test_size=0.25, random_state=42)

  def train_and_test(self, train_set, train_labels, test_set, test_labels, neighbor_count, job_count):
    classifier = KNeighborsClassifier(n_neighbors=neighbor_count, n_jobs=job_count)
    classifier.fit(train_set, train_labels)
    return classifier.score(test_set, test_labels)

  #retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
  def image_to_feature_vector(self, image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

  #retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
  def extract_color_histogram(self, image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
      [0, 180, 0, 256, 0, 256])
   
    #normalize
    cv2.normalize(hist, hist)
   
    # return the flattened histogram as the feature vector
    return hist.flatten()