# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import logging
import argparse

#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

##Global Functions
def print_to_console_log(str1):
    print(str1)
    logging.debug(str1)
    
def evaluate_prediction(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total

##Global Constants
DIRECTORY = "SVM_logs/"
TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S_')
FILE_NAME = DIRECTORY + TIME_STAMP+'SMV_classifier.sav'

if __name__=='__main__':
   
    ##Read command line arguments    
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--c_const",required=False, default=50, \
                                    help= "SVM c parameter")
    ap.add_argument("-g","--gamma",required=False, default=200e-6,\
                                    help="Kernel SVM rbf gamma parameter")
    ap.add_argument("-t","--test_size",required=False,default=0.5,\
                                    help="Value between 0,1 indicating the size of the test set")
    args = vars(ap.parse_args())
    
    C = float(args['c_const'])
    GAMMA = float(args['gamma'])
    TEST_SIZE = float(args['test_size'])

    
    #Log file
    logging.basicConfig(
                filename=DIRECTORY+TIME_STAMP+"training.log",
                level=logging.DEBUG,
                format="%(asctime)s:%(message)s")
    
    print_to_console_log("SVM Constants \nGAMMA:{:012.4E} \nC    :{:08.4f} \n".format(GAMMA,C))
    
    # Importing the dataset
    X = pickle.load(open("codify_data.sav", 'rb'))
    dataset = pd.read_csv('exercise_data.csv',sep='>')
    y = dataset.iloc[:, 0].values
    
    # Split the dataset into the testset and the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = 0)
    datasetSize = y.shape[0]
    trainingSize = y_train.shape[0]
    testSize = y_test.shape[0]
    data_info = "Dataset Info\nDataset Size:{:6d}\nTrainingset Size:{:6d}\nTestset Size:{:6d}\nTest Proportion:{:.3f}\n".format\
                                        (datasetSize,trainingSize,testSize,TEST_SIZE)
    print_to_console_log(data_info)
    
    # Fitting Kernel SVM to the Training set
    startTime1 = time.time()
    classifier = SVC(C=C, gamma=GAMMA , kernel ='rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    endTime1 = time.time()
    
    formatedTime = time.strftime('%H:%M:%S', time.gmtime(endTime1-startTime1))
    print_to_console_log("Training Time\nTime:{:010.4f}s\nTime:{:s}\n".format(endTime1-startTime1,formatedTime))
    
    # save the model to disk
    pickle.dump(classifier, open(FILE_NAME, 'wb'))
    
    # Model Evaluation
    print_to_console_log("Evaluate Model...\n")
    startTime2 = time.time()
    y_predicted = classifier.predict(X_test[:,:])
    endTime2 = time.time()
    
    formatedTime = time.strftime('%H:%M:%S', time.gmtime(endTime2-startTime2))
    print_to_console_log("Evaluation Time\nTime:{:010.4f}s\nTime:{:s}\n".format(endTime2-startTime2,formatedTime))
    
    # Accuracy and Confusion Matrix
    print_to_console_log("Results...\n")
    
    cm = confusion_matrix(y_test[:], y_predicted)
    testEvaluation = evaluate_prediction(y_test[:],y_predicted)
    cm_text ="Confusion Matrix\n{:6d} {:6d}\n{:s}\n{:6d} {:6d}\nTest Samples:{:6d}\n".format(\
                                        cm[0,0],cm[0,1],"-"*20,cm[1,0],cm[1,1],y_test.shape[0])
    
    print_to_console_log(cm_text)
    print_to_console_log("Results\nAccuracy in test set: {:8.4f}".format(testEvaluation))

    #Make Result file
    f1 = DIRECTORY+TIME_STAMP+"SVM_C={:0.2f}_GAMMA={:6.2E}_TEST_SIZE={:.3f}_ACC={:0.4f}.txt".format(C,GAMMA,TEST_SIZE,testEvaluation)
    
    with open(f1,"w") as file:
        file.write("Results")