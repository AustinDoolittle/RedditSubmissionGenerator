import argparse as ap
import sys
import config
import secrets
import re
import os
from redditretriever.retriever import SubmissionRetriever
from classifiers.knn.knnclassifier import kNNClassifier
from classifiers.cnn.networks.lenet import LeNet
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from processing.processor import ImgProcessor 
from processing.processor import ProcessType
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle as pkl

def main(argv):
  #parse arguments
  parser = ap.ArgumentParser(description="This application retrieves submissions from Reddit using the Reddit API")
  parser.add_argument("--classes", default=config.DEF_CLASS_FILE, help="The file to load the list of classes from (Default = '" + config.DEF_CLASS_FILE + "'')")
  parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Increase verbosity of the application")
  parser.add_argument("--neighbors", default=config.DEF_NEIGHBORS, type=int, help="The number of neighbors to observe when classifying (Default = '" + str(config.DEF_NEIGHBORS) + "')")
  parser.add_argument("--jobs", default=config.DEF_JOBS, type=int, help="The number of cores to run the classification processes (Default = " + str(config.DEF_JOBS) + ")")
  parser.add_argument("--subcount", default=config.DEF_SUB_COUNT, type=int, help="The amount of submissions from each subreddit to pull (Default = " + str(config.DEF_SUB_COUNT) + ")")
  parser.add_argument("--knnhist", default=False, action='store_true', help="Use kNN classification by color histogram")
  parser.add_argument("--knnfeat", default=False, action='store_true', help="Use kNN classification by feature vector")
  parser.add_argument("--lenet", default=False, action="store_true", help="Use a CNN in the LeNet configuration")
  parser.add_argument("--channels", default=config.DEF_CHANNEL_COUNT, help="The count of target channels in the datasets")
  args = parser.parse_args(argv)

  args.verbose

  if args.lenet:
    class_type = "lenet"
  elif args.knnhist:
    class_type = "knnhist"
  elif args.knnfeat:
    class_type = "knn_feat"

  raw_class_filename = os.path.splitext(os.path.basename(args.classes))[0]
  pkl_file = config.PKL_CACHE_DIR + raw_class_filename + "_" + str(args.subcount) + "_" + class_type + ".pkl"
  get_data = True

  #attempt to open the file
  try:
    with open(pkl_file, 'rb') as f:
      cache = pkl.load(f)
      train_s = cache[0]
      test_s = cache[1]
      train_l = cache[2]
      test_l = cache[3]
      sr_count = cache[4]
      get_data = False
      print "\tRetrieved from Cache\n"
  except Exception as ex:
    #the file did not exist or could not be opened, we'll get data from Reddit
    print "\tNo cached data, retrieving from Reddit API \n"

  if get_data:
    #setup reddit object
    reddit = SubmissionRetriever(secrets.CLIENT_ID, 
                                secrets.CLIENT_SECRET, 
                                secrets.CLIENT_PASSWORD,
                                secrets.CLIENT_USER_AGENT,
                                secrets.CLIENT_USERNAME,
                                filename=args.classes,
                                verbose=args.verbose)
    submissions = reddit.get_submissions(args.subcount)
    sr_count = len(reddit.subreddits)
    batch_size = 16

    if args.lenet:
      train_s, test_s, train_l, test_l = ImgProcessor.get_conv_dataset(submissions, args.channels, args.verbose)
    elif args.knnhist:
      train_s, test_s, train_l, test_l = ImgProcessor.get_hist_dataset(submissions, args.verbose)
    elif args.knnfeat:
      train_s, test_s, train_l, test_l = ImgProcessor.get_feat_dataset(submissions, args.verbose)

    cache = (train_s, test_s, train_l, test_l, sr_count)

    print "Saving Cache to " + pkl_file
    try:
      with open(pkl_file, 'wb') as f:
        pkl.dump(cache, f, pkl.HIGHEST_PROTOCOL)
    except Exception as ex:
      print "\t~Unable to save submissions to cache: " + str(ex)


  if args.knnfeat or args.knnhist:
    knn_classifier = kNNClassifier(args.verbose)
    if args.knnhist:

      #Test classification accuracy
      print "Training and testing feature vector data"
      feat_acc = knn_classifier.train_and_test(train_feats, train_feat_labels, test_feats, test_feat_labels, args.neighbors, args.jobs)

      print "\n~~FINISHED~~"
      print "Feature Accuracy: " + str(feat_acc * 100) + "%\n"
    
    if args.knnfeat:
      print "Training and testing histure vector data"
      hist_acc = knn_classifier.train_and_test(train_hists, train_hist_labels, test_hists, test_hist_labels, args.neighbors, args.jobs)

      print "\n~~FINISHED~~"
      print "Histogram Accuracy: " + str(hist_acc * 100) + "%\n"

  elif args.lenet:

    net = LeNet(args.channels, 150, 150, sr_count)

    net.model.fit(
        np.asarray(train_s),
        np.asarray(train_l),
        validation_split=0.25,
        epochs=50)

    loss, accuracy = net.model.evaluate(np.asarray(test_s), np.asarray(test_l), verbose=1)

    print "loss: " + str(loss) + ", Accuracy: " + str(accuracy)

  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files if need be
  sys.exit(main(sys.argv[1:]))


