##General Libraries
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import logging
import argparse

##Sklearn Toolbox
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##Shogun Toolbox
from shogun import StringCharFeatures, RAWBYTE
from shogun import BinaryLabels
from shogun import SubsequenceStringKernel
from shogun import LibSVM


##Global Functions
def print_to_console_log(str1):
    print(str1)
    logging.debug(str1)
    
def cleanText(text):
    filterCharacters = "_\'*;!%&$#()-0123456789.,/?:'\""
    cleanText = ''.join(list(filter(lambda ch: ch not in filterCharacters,text)))
    return cleanText

def evaluate_prediction(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total


##Global Constants
DIRECTORY = "SVM_logs_kernel/"
TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S_')
FILE_NAME = DIRECTORY + TIME_STAMP+'SMV_classifier_kernel_ssk.sav'


if __name__=='__main__':
   
    ## Read command line arguments    
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--c_const",required=False, default=100, \
                                    help= "SVM c parameter")
    ap.add_argument("-l","--lambda",required=False, default=0.7,\
                                    help="String subsequence kernel lambda parameter(value between 0 and 1)")
    ap.add_argument("-s", "--subsequence_size", required=False, default=5, \
                                    help="integer value indicating the the maximum size of the subsequence")
    ap.add_argument("-t", "--test_size", required=False, default=0.5, \
                                    help="Value between 0,1 indicating the size of the test set")
    ap.add_argument("-r", "--training_size", required=False, default=1000, \
                                    help="integer value indicating the training size")
    args = vars(ap.parse_args())
    
    ## Model parameters
    C = float(args['c_const'])
    LAMBDA = float(args['lambda'])
    SUBSEQUENCE_SIZE = int(args['subsequence_size'])
    TEST_SIZE = float(args['test_size'])
    TRAINING_SIZE = int(args['training_size'])

    ## Log file
    logging.basicConfig(
                filename=DIRECTORY+TIME_STAMP+"training.log",
                level=logging.DEBUG,
                format="%(asctime)s:%(message)s")
   
    ## Importing the dataset
    dataset = pd.read_csv('exercise_data.csv',sep='>')
    X_raw = dataset.iloc[:, 2].values.tolist()
    y = dataset.iloc[:, 0].values
    
    ## Cleaning texts
    for i in range(len(y)):
        text = cleanText(X_raw[i])
        X_raw[i] = text
    
    X = X_raw

    ## Print Model Constants
    print_to_console_log\
        ("SVM Constants \nLAMBDA:{:0.2E} \nC    :{:0.2f} \nSUBSEQUENCE: {:d}\n".format(LAMBDA,C,SUBSEQUENCE_SIZE))

    ## Split the dataset into the testset and the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = 0)

    X_train = X_train[:TRAINING_SIZE]
    X_test = X_test[:TRAINING_SIZE]
    y_test = y_test[:TRAINING_SIZE]
    y_train = y_train[:TRAINING_SIZE]

    datasetSize = y.shape[0]
    trainingSize = y_train.shape[0]
    testSize = y_test.shape[0]
    data_info = "Dataset Info\nDataset Size:{:6d}\nTrainingset Size:{:6d}\nTestset Size:{:6d}\nTest Proportion:{:.3f}\n".format\
                                                    (datasetSize,trainingSize,testSize,TEST_SIZE)
    print_to_console_log(data_info)

    ## Codify Labels (1,-1)
    y_train[np.where(y_train==0)] = -1
    y_test [np.where(y_test==0)]  = -1
    
    ## Set up Subsequence string Kernel
    features = StringCharFeatures(X_train, RAWBYTE)
    test_features = StringCharFeatures(X_test, RAWBYTE)
    labels = BinaryLabels(y_train)
    sk = SubsequenceStringKernel(features, features, SUBSEQUENCE_SIZE, LAMBDA)

    ## Fitting Kernel SVM to the Training set
    startTime1 = time.time()
    svm = LibSVM(C, sk, labels)
    svm.train()
    endTime1 = time.time()

    ## Print Training Time
    formatedTime = time.strftime('%H:%M:%S', time.gmtime(endTime1-startTime1))
    print_to_console_log("Training Time\nTime:{:010.4f}s\nTime:{:s}\n".format(endTime1-startTime1,formatedTime))

    # save the model to disk
    pickle.dump(svm, open(FILE_NAME, 'wb'))

    # Model Evaluation
    print_to_console_log("Evaluate Model...\n")
    startTime2 = time.time()
    predicted_labels = svm.apply(test_features).get_labels()
    y_predicted = np.array(predicted_labels)
    endTime2 = time.time()

    ## Print Evaluation Time
    formatedTime = time.strftime('%H:%M:%S', time.gmtime(endTime2-startTime2))
    print_to_console_log("Evaluation Time\nTime:{:010.4f}s\nTime:{:s}\n".format(endTime2-startTime2,formatedTime))

    ## Accuracy and Confusion Matrix
    print_to_console_log("Results...\n")

    cm = confusion_matrix(y_test[:], y_predicted)
    testEvaluation = evaluate_prediction(y_test[:],y_predicted)
    cm_text ="Confusion Matrix\n{:6d} {:6d}\n{:s}\n{:6d} {:6d}\nTest Samples:{:6d}\n".format(\
                                        cm[0,0],cm[0,1],"-"*20,cm[1,0],cm[1,1],y_test.shape[0])

    print_to_console_log(cm_text)
    print_to_console_log("Results\nAccuracy in test set: {:8.4f}".format(testEvaluation))

    #Make Result file
    f1 = DIRECTORY+TIME_STAMP+"SVM_C={:0.2f}_LAMBDA={:6.2E}_TRAINING_SIZE={:.3f}_ACC={:0.4f}_SUB_SIZE={:d}.txt".format(C,LAMBDA,TRAINING_SIZE,testEvaluation,SUBSEQUENCE_SIZE)

    with open(f1,"w") as file:
        file.write("Results\n")
