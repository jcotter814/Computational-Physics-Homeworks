# -*- coding: utf-8 -*-
"""
hw8problem4.py
@author: James Cotter
Convolutional neural network to recognize the shape of galaxies as elliptical,
spiral or irregular.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np

#Import data
with np.load('galaxy.npz') as f:
    X = f['X']
    Y = f['Y']

#Shuffle Data
length = len(Y)
shuffle_index = np.random.permutation(length)
rand_x = X[shuffle_index]
rand_y = Y[shuffle_index]

#Create training and test data
trainLength = int(0.8*len(Y))
x_train, x_test, y_train, y_test = rand_x[:trainLength], rand_x[trainLength:], rand_y[:trainLength], rand_y[trainLength:]

#shape
x_train = x_train[:,:,:,np.newaxis] / 255.0
x_test = x_test[:,:,:,np.newaxis] / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# build the system
model4 = Sequential()
model4.add(Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', input_shape=(255,255,1)))
model4.add(MaxPooling2D(pool_size=3))
model4.add(Flatten())
model4.add(Dense(3, activation='softmax'))

#compile
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#use
model4.fit(x_train, y_train, epochs=5, validation_split=0.1)

#test accuracy
_, test_acc = model4.evaluate(x_test, y_test)
print("Accuracy is: ", test_acc)


