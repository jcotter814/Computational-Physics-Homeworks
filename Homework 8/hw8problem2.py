# -*- coding: utf-8 -*-
"""
hw8problem2.py
Created on Sun Dec  8 18:07:34 2019
@author: James Cotter
Classification with data augmentation.
"""
#Imports
import numpy as np
import os

import sklearn
print("Scikit Learn version: ", sklearn.__version__)

import warnings
from sklearn.exceptions import ConvergenceWarning

#Ignore warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

#import data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target
y = y.astype(np.int)


#plotting stuff
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#example digit
some_digit = X[4200] 
some_digit_image = some_digit.reshape(28, 28) # reshape for visualization

#Horizontal shift
def horiz_shift(image,shift,val):
    """Shifts the matrix along the x axis shift times and pads the shifted 
    values with val."""
    r = np.empty_like(image)
    if shift > 0:
        r[:,:shift] = val
        r[:,shift:] = image[:,:-shift]
    elif shift < 0:
        r[:,shift:] = val
        r[:,:shift] = image[:,-shift:]
    else:
        return image
    return r

#Vertical shift
def vert_shift(image,shift,val):
    """Shifts the matrix along the x axis shift times and pads the shifted 
    values with val."""
    r = np.empty_like(image)
    if shift > 0:
        r[:shift,:] = val
        r[shift:,:] = image[:-shift,:]
    elif shift< 0:
        r[shift:,:] = val
        r[:shift,:] = image[-shift:,:]
    else:
        return image
    return r

#---------------Part a-------------------
fig,axs = plt.subplots(1,5,figsize=(7,7))

#original
axs[0].imshow(some_digit_image, cmap = plt.get_cmap('gray_r'))
axs[0].set_title('Original')

#positive horizontal shift
h1 = horiz_shift(some_digit_image,3,0)
axs[1].imshow(h1, cmap = plt.get_cmap('gray_r'))
axs[1].set_title('+ Horizontal Shift')

#negative horizontal shift
h2 = horiz_shift(some_digit_image,-3,0)
axs[2].imshow(h2, cmap = plt.get_cmap('gray_r'))
axs[2].set_title('- Horizontal Shift')

#positive vertical shift
v1 = vert_shift(some_digit_image,3,0)
axs[3].imshow(v1, cmap = plt.get_cmap('gray_r'))
axs[3].set_title('+ Vertical Shift')

#negative vertcial shift
v2 = vert_shift(some_digit_image,-3,0)
axs[4].imshow(v2, cmap = plt.get_cmap('gray_r'))
axs[4].set_title('- Vertical Shift')

plt.tight_layout()

#---------Part b (unaugmented data)------------------------

# separate the dataset to training and testing sets
trainLength = 55000
X_train, X_test, y_train, y_test = X[:trainLength], X[trainLength:], y[:trainLength], y[trainLength:]

# randomize the order of the training set
np.random.seed(42)  # consistent across runs
shuffle_index = np.random.permutation(trainLength)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

#cross val predict
#from sklearn.model_selection import cross_val_predict
#y_train_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3)

forest_clf.fit(X_train,y_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
CM1 = confusion_matrix(y_train, forest_clf.predict(X_train))
print("Confusion matrix of unaugmented data: ",CM1) # rows: actual classes, column: predicted classes

#accuracy score
from sklearn.metrics import accuracy_score
forest_acc = accuracy_score(y_test,forest_clf.predict(X_test))
print('Accuracy of unaugmented data: ',forest_acc)

#----------part b (augmented data)----------------

val = 0
shift = 1

#Convert to 3d
three_d = X_train.reshape(-1,28,28)

#positive vertical shift and append
ph = np.empty_like(three_d)
ph[:,:,:shift] = val
ph[:,:,shift:] = three_d[:,:,:-shift]
X_train_augmented = np.append(three_d,ph,axis = 0)
y_train = np.append(y_train,y_train) #add correspondig values

#negative horizontal shift and append
nh = np.empty_like(three_d)
nh[:,:,shift:] = val
nh[:,:,:shift] = three_d[:,:,-shift:]
X_train_augmented = np.append(X_train_augmented,nh,axis = 0)
y_train = np.append(y_train,y_train) #add correspondig values

#positive vertical shift and append
pv = np.empty_like(three_d)
pv[:,:shift,:] = val
pv[:,shift:,:] = three_d[:,:-shift,:]
X_train_augmented = np.append(X_train_augmented,pv,axis = 0)
y_train = np.append(y_train,y_train) #add correspondig values

#negative vertical shift and append
nv = np.empty_like(three_d)
nv[:,shift:,:] = val
nv[:,:shift,:] = three_d[:,-shift:,:]
X_train_augmented = np.append(X_train_augmented,nv, axis = 0)
y_train = np.append(y_train,y_train) #add correspondig values

#reshape back to original form
augmented_data = X_train_augmented.reshape(-1,784)

# randomize the order of the training set
np.random.seed(42)  # consistent across runs
shuffle_index = np.random.permutation(trainLength)
X_train_augmented, y_train = X_train[shuffle_index], y_train[shuffle_index]

#augmented
aug_forest_clf = RandomForestClassifier(random_state=42)

aug_forest_clf.fit(X_train_augmented,y_train)

#confusion matrix
CM1_augmented = confusion_matrix(y_train, aug_forest_clf.predict(X_train_augmented))

#accuracy score
forest_acc_augmented = accuracy_score(y_test,aug_forest_clf.predict(X_test))
print('Accuracy of augmented data: ',forest_acc_augmented)


#Compare accuracy
print("Accuracy difference of: ",forest_acc-forest_acc_augmented)


#Plot confusion matrices
#normalize
row_sums = CM1.sum(axis = 1,keepdims = True)
aug_row_sums = CM1_augmented.sum(axis=1, keepdims=True)

aug_norm_conf_mx = CM1_augmented / aug_row_sums
norm_conf_mx = CM1/row_sums

# zero out the diagonal
np.fill_diagonal(aug_norm_conf_mx, 0)
np.fill_diagonal(norm_conf_mx,0)

fig = plt.figure(2)
fig.add_subplot(1,2,1)
plt.matshow(aug_norm_conf_mx, cmap=plt.cm.gray,fignum = False)
plt.title('Augmented Data Set')

fig.add_subplot(1,2,2)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray,fignum = False)
plt.title('Regular Data Set')

plt.tight_layout()
plt.show()









