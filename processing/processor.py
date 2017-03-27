from sklearn.cross_validation import train_test_split
import urllib
import numpy as np
from enum import Enum
import cv2

class ProcessType(Enum):
  FeatureVector = 0
  ColorHistogram = 1
  Convolutional = 2

class ImgProcessor(object):

  @staticmethod
  def url_to_cv_image( url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

  @staticmethod
  def url_to_conv_image(url):
    resp = urllib.urlopen(url)
    return np.asarray(bytearray(resp.read()), dtype="uint8")

  #retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
  @staticmethod
  def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

  #retrieved from http://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
  @staticmethod
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

  @staticmethod
  def convert_to_dataset(submissions, verbose=False, type=ProcessType.ColorHistogram):
    retval = []
    classes = []

    #iterate over subreddits
    for subreddit in submissions:
      counter = 1
      total_count = len(submissions[subreddit])
      if verbose:
        print "\tProcessing images from r/" + subreddit

      #iterate over submissions in subreddit
      for submission in submissions[subreddit]:
        if type == ProcessType.Convolutional:
          try:
            conv_img = ImgProcessor.url_to_conv_image(submission['link'])
          except (IOError):
            print "\t\t~Error Retreiving image, skipping..."
            retval.append(conv_img)
            classes.append(subreddit)

        else:
          try:
            #Get image
            cv_img = ImgProcessor.url_to_cv_image(submission['link'])
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
            dat = ImgProcessor.image_to_feature_vector(cv_img)
          elif type == ProcessType.ColorHistogram:
            dat = ImgProcessor.extract_color_histogram(cv_img)
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

        if verbose:
          print "\t\tr/" + subreddit + ": " + str(counter) + "/" + str(total_count)
        counter += 1

    return train_test_split(retval, classes, test_size=0.25, random_state=42)
