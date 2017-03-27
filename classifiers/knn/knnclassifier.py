import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum
import cv2



class kNNClassifier(object):
  def __init__(self, verbose=False):
    self.verbose = verbose

  def train_and_test(self, train_set, train_labels, test_set, test_labels, neighbor_count, job_count):
    classifier = KNeighborsClassifier(n_neighbors=neighbor_count, n_jobs=job_count)
    classifier.fit(train_set, train_labels)
    return classifier.score(test_set, test_labels)