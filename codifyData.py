# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from collections import defaultdict
import pickle


def cleanSplitText(text):
    filterCharacters = "_\'*;!%&$#()-0123456789.,/?:'\""
    cleanText = ''.join(list(filter(lambda ch: ch not in filterCharacters,text)))
    cleanText = cleanText.split(' ')
    return cleanText

def encodeFeatures(text,bow):
    pass

# Importing the dataset
dataset = pd.read_csv('exercise_data.csv',sep='>')
X_raw = dataset.iloc[:, 2].values.tolist()
y = dataset.iloc[:, 0].values.tolist()

#Read bag of Words
bagOfWords = defaultdict(lambda : -10)
with open("bow.txt","r") as bow:
    words=[ line.strip("\n").split(',') for line in bow]
    for (key, value) in words: 
        bagOfWords[key]=int(value)
    

datasetSize = len(y)
featuresSize = len(bagOfWords)
X = np.zeros((datasetSize,featuresSize),dtype=np.uint8)

print("Start Encoding")
for i in range(datasetSize):
    cleanText = cleanSplitText(X_raw[i])
    textFeat = np.zeros((1,featuresSize))
    for word in cleanText:
        j = bagOfWords[word]
        if len(word)>3 and word[-1]=='s':
                word=word[0:-1]
        if (j>=0):
            textFeat[0,j] +=1
    X[i,:] = textFeat
    if i%1000==0:
        print(i)

print("Saving Features") 
# save the model to text file
#np.savetxt("codify_data.txt",X,fmt='%u', delimiter=',')
# save the model to binary file
pickle.dump(X, open("codify_data.sav", 'wb'))

print("Finish Encoding Data")
