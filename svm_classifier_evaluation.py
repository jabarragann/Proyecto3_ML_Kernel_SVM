# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def evaluate_prediction(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total

#import data set
X = pickle.load(open("codify_data.sav", 'rb'))
dataset = pd.read_csv('exercise_data.csv',sep='>')
y = dataset.iloc[:, 0].values

# Split the dataset into the testset and the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)

#Load Classifier
print("Loading classifier...")
classifier = pickle.load(open("SVM_logs/05-06_23-29-10_SMV_classifier.sav", 'rb'))

# Predicting the Test set results
print("Making predictions...")
y_predicted = classifier.predict(X_test[:,:])

# Making the Confusion Matrix
print("Results...")
cm = confusion_matrix(y_test[:], y_predicted)
print(cm)
testEvaluation = evaluate_prediction(y_test[:],y_predicted)
print("Accuracy in test set: {:08.4f}".format(testEvaluation))
