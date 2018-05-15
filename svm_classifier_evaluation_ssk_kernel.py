# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

##Shogun Toolbox
from shogun import StringCharFeatures, RAWBYTE
from shogun import BinaryLabels
from shogun import SubsequenceStringKernel
from shogun import LibSVM


def evaluate_prediction(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total

def cleanText(text):
    filterCharacters = "_\'*;!%&$#()-0123456789.,/?:'\""
    cleanText = ''.join(list(filter(lambda ch: ch not in filterCharacters,text)))
    return cleanText


## Importing the dataset
dataset = pd.read_csv('exercise_data.csv', sep='>')
X_raw = dataset.iloc[:, 2].values.tolist()
y = dataset.iloc[:, 0].values

## Cleaning texts
for i in range(len(y)):
    text = cleanText(X_raw[i])
    X_raw[i] = text

X = X_raw

# Split the dataset into the testset and the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)

X_test = X_test[:10]
X_test = StringCharFeatures(X_test, RAWBYTE)
y_test = y_test[:10]
y_test [np.where(y_test==0)]  = -1

#Load Classifier
print("Loading classifier...")
svm = pickle.load(open("SVM_logs_kernel/05-11_00-44-27_SMV_classifier.sav", 'rb'))

# Predicting the Test set results
print("Making predictions...")
y_predicted = svm.apply(X_test).get_labels()
y_predicted = np.array(y_predicted)

# Making the Confusion Matrix
print("Results...")
cm = confusion_matrix(y_test[:], y_predicted)
print(cm)
testEvaluation = evaluate_prediction(y_test[:],y_predicted)
print("Accuracy in test set: {:08.4f}".format(testEvaluation))
