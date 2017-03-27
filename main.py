import argparse as ap
import sys
import config
import secrets
from redditretriever.retriever import SubmissionRetriever
from classifiers.knn.knnclassifier import kNNClassifier
from classifiers.cnn.networks.lenet import LeNet
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from processing.processor import ImgProcessor 
from processing.processor import ProcessType

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
  args = parser.parse_args(argv)

  args.verbose

  print "Creating SubmissionRetriever object"

  #setup reddit object
  reddit = SubmissionRetriever(secrets.CLIENT_ID, 
                                  secrets.CLIENT_SECRET, 
                                  secrets.CLIENT_PASSWORD,
                                  secrets.CLIENT_USER_AGENT,
                                  secrets.CLIENT_USERNAME,
                                  filename=args.classes,
                                  verbose=args.verbose)


  #get submissions
  submissions = reddit.get_submissions(args.subcount)

  if args.knnfeat or args.knnhist:
    knn_classifier = kNNClassifier(args.verbose)
    if args.knnhist:
      #Get Training and Test feature vector sets
      print "Shuffling and creating histogram training and test set"
      train_feats, test_feats, train_feat_labels, test_feat_labels = ImgProcessor.convert_to_dataset(submissions, verbose=args.verbose, type=ProcessType.FeatureVector)

      #Test classification accuracy
      print "Training and testing feature vector data"
      feat_acc = knn_classifier.train_and_test(train_feats, train_feat_labels, test_feats, test_feat_labels, args.neighbors, args.jobs)

      print "\n~~FINISHED~~"
      print "Feature Accuracy: " + str(feat_acc * 100) + "%\n"
    
    if args.knnfeat:
      #Get Training and Test feature vector sets
      print "Shuffling and creating histogram training and test set"
      train_hists, test_hists, train_hist_labels, test_hist_labels = ImgProcessor.convert_to_dataset(submissions, verbose=args.verbose, type=ProcessType.ColorHistogram)

      #Test classification accuracy
      print "Training and testing histure vector data"
      hist_acc = knn_classifier.train_and_test(train_hists, train_hist_labels, test_hists, test_hist_labels, args.neighbors, args.jobs)

      print "\n~~FINISHED~~"
      print "Histogram Accuracy: " + str(hist_acc * 100) + "%\n"

  elif args.lenet:
    #code augmented from http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

    train_set, test_set, train_labels, test_labels = ImgProcessor.convert_to_dataset(submissions, verbose=args.verbose, type=ProcessType.Convolutional)

    # transform the training and testing labels into vectors in the
    # range [0, classes] -- this generates a vector for each label,
    # where the index of the label is set to `1` and all other entries
    # to `0`; in the case of MNIST, there are 10 class labels
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet(3, len(reddit.subreddits))
    model.compile("categorical_crossentropy", opt, ["accuracy"])

    print "Training..."
    model.fit(trainData, trainLabels, 128, 20, 1)
   
    # show the accuracy on the testing set
    print "Testing..."
    (loss, accuracy) = model.evaluate(testData, testLabels,
      batch_size=128, verbose=1)
    print("DONE, Accuracy: {:.2f}%".format(accuracy * 100))


  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files if need be
  sys.exit(main(sys.argv[1:]))


