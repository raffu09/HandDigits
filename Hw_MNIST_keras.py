# python Hw_MNIST_keras.py
# no arguments
# keras version by Marko.Rantala@gmail.com
# v1.01 20170410


# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf

# first version 
#import tflearn
#import tflearn.datasets.mnist as mnist  ## 

#
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils

#mnist is in keras  too !
from keras.datasets import mnist as mndata

from matplotlib import pyplot as plt

import sys


# Retrieve the training and test data
# trainX, trainY, testX, testY = mnist.load_data(one_hot=True)


# print(trainX.shape)
# print(trainX)

# print(trainY.shape)
# print(trainY)

# print("another format: data amount seems to be different than TFLearn version  ")
(trainX, trainY), (testX, testY) = mndata.load_data()

print(trainX.shape)
print(trainX)

print(trainY.shape)
print(trainY)

plt.imshow(trainX[1])
plt.show()

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

#lets shape  trainY and testY to the  one_hot model
# there are probably smarter ways to reshape this to one_hot model in single line... 
def one_hot(number):
    a = np.zeros(10, dtype=np.int8)
    a[number] = 1
    return a
 
# tmp = np.zeros((len(trainY), 10)) #dtype=np.int8) should it have that
# for i in range(len(trainY)):
    # tmp[i, trainY[i]] = 1

# trainY = tmp

# tmp = np.zeros((len(testY), 10)) #dtype=np.int8) should it have that
# for i in range(len(testY)):
    # tmp[i, testY[i]] = 1

# testY = tmp

trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)


print(trainY.shape)
print(trainY)
 
#to 3 D, why this is needed?
#trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1]) # or trainY.shape[0], trainY[1], 1 ??????
trainX2D = trainX.reshape(trainX.shape[0],  trainX.shape[1], trainX.shape[2], 1)
testX2D = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)
    
trainX2D.shape
  

# Visualizing the data
import matplotlib.pyplot as plt
#%matplotlib inline # no more jupyter


# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = trainY[index].argmax(axis=0)
    #label = -1
    # Reshape 784 array into 28x28 image
    #image = trainX[index].reshape([28,28])
    image = trainX[index]
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
#reshape training data back to array of "images" so use it as a input (convolution models...)


    
# Display the first (index 0) training image
show_digit(0)


# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
   
    # Include the input layer, hidden layer(s), and set how you want to train the model
    # net = tflearn.input_data([None, trainX.shape[1]])
    
    # net = tflearn.fully_connected(net, 127, activation='ReLU')
    
    # net = tflearn.fully_connected(net, 63, activation='ReLU')
    
    # net = tflearn.fully_connected(net, trainY.shape[1], activation='softmax')
    
    # net = tflearn.regression(net, optimizer='sgd', learning_rate=0.07, loss='categorical_crossentropy')
    
    #This model assumes that your network is named "net"    
    # model = tflearn.DNN(net)
    
    # previous turned to the keras, quite similar but even simpler
    # NO more simpler as convolutional layer added, BTW: here you could remove the next Dense layer
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=( trainX.shape[1], trainX.shape[2], 1), kernel_initializer='uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(255,  activation='relu'))
    
    model.add(Dense(127,  activation='relu'))
    #model.add(Dense(63, activation='relu'))
    model.add(Dense(trainY.shape[1], activation='softmax'))
    
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
   
    
    return model
    
    
# Build the model
model = build_model()  # 

# this is a nice part, how to get tensorboard data!
callbacksInitial = [ EarlyStopping(monitor='loss', patience=128), TensorBoard(log_dir='../Logs/'+sys.argv[0][:-3], histogram_freq=20, write_graph=True, write_images=False)   ]  # val_loss as example
#
# start tensorboard later with command
#tensorboard --logdir=full path to previous log dir,  named to this program
# Training
# TFlearn: model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=100)

model.fit(trainX2D, trainY,  validation_split=0.1, batch_size=200, epochs=300, callbacks=callbacksInitial)


# Compare the labels that our model predicts with the actual labels

# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = np.array(model.predict(testX2D)).argmax(axis=1)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)


