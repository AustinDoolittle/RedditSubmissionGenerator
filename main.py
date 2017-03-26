import argparse as ap
import sys
import config
import secrets
import redditretriever.retriever as rr
import classifiers.knn.knnclassifier as knn

def main(argv):
  #parse arguments
  parser = ap.ArgumentParser(description="This application retrieves submissions from Reddit using the Reddit API")
  parser.add_argument("--classes", default=config.DEF_CLASS_FILE, help="The file to load the list of classes from (Default = '" + config.DEF_CLASS_FILE + "'')")
  parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Increase verbosity of the application")
  parser.add_argument("--neighbors", default=config.DEF_NEIGHBORS, type=int, help="The number of neighbors to observe when classifying (Default = '" + str(config.DEF_NEIGHBORS) + "')")
  parser.add_argument("--jobs", default=config.DEF_JOBS, type=int, help="The number of cores to run the classification processes (Default = " + str(config.DEF_JOBS) + ")")
  parser.add_argument("--subcount", default=config.DEF_SUB_COUNT, type=int, help="The amount of submissions from each subreddit to pull (Default = " + str(config.DEF_SUB_COUNT) + ")")
  args = parser.parse_args(argv)

  args.verbose

  print "Creating SubmissionRetriever object"

  #setup reddit object
  reddit = rr.SubmissionRetriever(secrets.CLIENT_ID, 
                                  secrets.CLIENT_SECRET, 
                                  secrets.CLIENT_PASSWORD,
                                  secrets.CLIENT_USER_AGENT,
                                  secrets.CLIENT_USERNAME,
                                  filename=args.classes,
                                  verbose=args.verbose)


  #get submissions
  submissions = reddit.get_submissions(args.subcount)

  knn_classifier = knn.kNNClassifier(args.verbose)

  #Get Training and Test feature vector sets
  print "Shuffling and creating histogram training and test set"
  train_feats, test_feats, train_feat_labels, test_feat_labels = knn_classifier.convert_to_dataset(submissions, type=knn.ProcessType.FeatureVector)

  #Test classification accuracy
  print "Training and testing feature vector data"
  feat_acc = knn_classifier.train_and_test(train_feats, train_feat_labels, test_feats, test_feat_labels, args.neighbors, args.jobs)

   #Get Training and Test feature vector sets
  print "Shuffling and creating histogram training and test set"
  train_hists, test_hists, train_hist_labels, test_hist_labels = knn_classifier.convert_to_dataset(submissions, type=knn.ProcessType.ColorHistogram)

  #Test classification accuracy
  print "Training and testing histure vector data"
  hist_acc = knn_classifier.train_and_test(train_hists, train_hist_labels, test_hists, test_hist_labels, args.neighbors, args.jobs)



  #Display results
  print "~~FINISHED~~"
  print "Feature Accuracy: " + str(feat_acc * 100) + "%"
  print "Histogram Accuracy: " + str(hist_acc * 100) + "%"

  return 0

if __name__ == "__main__":
  #call the main method, do this so we can also call the main method from other files if need be
  sys.exit(main(sys.argv[1:]))


