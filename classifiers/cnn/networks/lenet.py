from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


 
#code augmented from http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
class LeNet:

  def __init__(self, depth, classes, weightsPath=None):
    # initialize the model
    self.model = Sequential()

    # first set of CONV => RELU => POOL
    self.model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(depth, None, None)))
    self.selfmodel.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    self.model.add(Convolution2D(50, 5, 5, border_mode="same"))
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    self.model.add(Flatten())
    self.model.add(Dense(500))
    self.model.add(Activation("relu"))
 
    # softmax classifier
    self.model.add(Dense(classes))
    self.model.add(Activation("softmax"))

    # if a weights path is supplied (inicating that the model was
    # pre-trained), then load the weights
    if weightsPath is not None:
      self.model.load_weights(weightsPath)

  def compile(self, loss, optimizer, metrics):
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  def fit(trainData, trainLabels, batch_size, nb_epoch, verbose):
    self.model.fit(trainData, trainLabels, batch_size, nb_epoch, verbose)

  def evaluate(testData, testLabels, batch_size, verbose):
    self.model.evaluate(testData, testLabels, batch_size=batch_size, verbose=verbose)

