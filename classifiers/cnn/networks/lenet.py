from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD


 
#code augmented from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
class LeNet:

  def __init__(self, depth, height, width, class_count):

    self.model = Sequential()

    self.model.add(Conv2D(32, (3, 3), input_shape=(depth, height, width), data_format='channels_first'))
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Conv2D(32, (3, 3)))
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Conv2D(64, (3, 3)))
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    self.model.add(Dense(64))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(1))
    self.model.add(Activation('sigmoid'))

    self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

