import urllib
import numpy as np
from enum import Enum
import cv2
import cStringIO
from PIL import Image
import sys
from keras.preprocessing.image import img_to_array, ImageDataGenerator

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
  def url_to_image(url):
    file = cStringIO.StringIO(urllib.urlopen(url).read())
    return Image.open(file)

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
  def get_hist_dataset(submissions, verbose=False):
    return ImgProcessor.knn_converter(submissions, ImgProcessor.extract_color_histogram)
    

  @staticmethod
  def get_feat_dataset(submissions, verbose=False):
    return ImgProcessor.knn_converter(submissions, ImgProcessor.image_to_feature_vector)

  @staticmethod
  def knn_converter(submissions, method_to_call, verbose=False):
    for subreddit in submissions:
      try:
        #Get image
        cv_img = ImgProcessor.url_to_cv_image(submissions['link'])
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

      #call passed method
      method_to_call(cv_img)

      #check for histogram error
      if dat is None:
        print "\t\t~Error Converting raw image to histogram, skipping..."
        total_count -= 1
        continue

      #append to return values
      retval.append(dat)
      classes.append(subreddit)

    return train_test_split(retval, classes, test_size=0.25, random_state=42)

  @staticmethod
  def get_conv_dataset(submissions, channels, verbose=False):
    retval = []
    classes = []

    subreddits = submissions.keys()

    #iterate over subreddits
    for subreddit in submissions:
      counter = 1
      total_count = len(submissions[subreddit])

      if verbose:
        print "\tProcessing images from r/" + subreddit

      #iterate over submissions
      for submission in submissions[subreddit]:
        try:
          img = ImgProcessor.url_to_image(submission['link'])
          img = img.resize((150,150))
        except IOError:
          print "\t~Error converting image, skipping"
          total_count -= 1
          continue

        img_arr = img_to_array(img)
        if img_arr.shape[2] != channels:
          print "\t~Does not have 3 channels, skipping"
          total_count -= 1
          continue

        if verbose:
          print "\t\tr/" + subreddit + " " + str(counter) + "/" + str(total_count)

        counter += 1

        #TODO: I don't like that we are reshaping the data 
        #instead of shaping the input of the CNN to fit the data
        img_arr = img_arr.reshape((img_arr.shape[2], img_arr.shape[0], img_arr.shape[1]))

        retval.append(img_arr)
        classes.append(subreddit)

    classes = [subreddits.index(x) for x in classes]
    train_s, test_s, train_l, test_l = train_test_split(retval, classes, test_size=.25)
    return np.asarray(train_s), np.asarray(test_s), np.asarray(train_l), np.asarray(test_l)
