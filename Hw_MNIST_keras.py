# python Hw_MNIST_keras.py
# no arguments
# keras version by Marko.Rantala@gmail.com
# v1.01 20170410


# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf

# first version 
import tflearn
import tflearn.datasets.mnist as mnist  ## 

#
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import sys


# Retrieve the training and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

print(trainX.shape)
print(trainX)

print(trainY.shape)
print(trainY)


# Visualizing the data
import matplotlib.pyplot as plt
#%matplotlib inline # no more jupyter

# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
# Display the first (index 0) training image
show_digit(0)

# this is a nice part, how to get tensorboard data!
callbacksInitial = [ EarlyStopping(monitor='loss', patience=128), TensorBoard(log_dir='../Logs/'+sys.argv[0][:-3], histogram_freq=40, write_graph=True, write_images=True)   ]  # val_loss as example

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
    model = Sequential()
    model.add(Dense(127, batch_input_shape=(None, trainX.shape[1]), init='uniform', activation='relu'))
    model.add(Dense(63, activation='relu'))
    model.add(Dense(trainY.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
   
    
    return model
    
    
# Build the model
model = build_model()  # 

# this is a nice part, how to get tensorboard data!
callbacksInitial = [ EarlyStopping(monitor='loss', patience=128), TensorBoard(log_dir='../Logs/'+sys.argv[0][:-3], histogram_freq=40, write_graph=True, write_images=True)   ]  # val_loss as example
# Training
# TFlearn: model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=100)

model.fit(trainX, trainY,  validation_split=0.1, batch_size=100, epochs=100, callbacks=callbacksInitial)


# Compare the labels that our model predicts with the actual labels

# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = np.array(model.predict(testX)).argmax(axis=1)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)


