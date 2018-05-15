# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:27:39 2018

@author: Juan Antonio Barragán Noguera
@email: jabarragann@unal.edu.co

"""

import pandas as pd
import numpy as np


#constants
classSize = 5000
exerciseDataFile = 'exercise_data.csv'
 
with open('news.csv') as newsFile:
    news = [line.strip().replace('"','') for line in newsFile]
    
with open('labels.csv') as labelsFile:
    labels =[line.strip().replace('"','') for line in labelsFile]
    
numberOfNews=len(news)
count0=0
count1=0

us_world = []
sport_entertainment = []

for label,story in zip(labels,news):
    if label =='us' or label=='world':
        us_world += [(label,story)]
    elif label=='sport' or label == 'entertainment':
        sport_entertainment += [(label,story)]

randIdx = np.random.choice(min(len(us_world),len(sport_entertainment)), classSize, replace=False) 
us_world = np.array(us_world)[randIdx,:]
sport_entertainment = np.array(sport_entertainment)[randIdx,:]

randIdx = np.random.choice(classSize*2,classSize*2,replace=False)
data = np.append(us_world,sport_entertainment,axis=0)
data = data[randIdx,:]

with open(exerciseDataFile,'w') as exerciseFile:
    exerciseFile.write("class>label>micro_news\n")
    
    for index in range(data.shape[0]):
        if  data[index,0]=='us' or data[index,0]=='world':
            exerciseFile.write("{:s}>{:s}>{:s} \n".format("1",data[index,0],data[index,1]))
            count1+=1
        elif data[index,0]=='sport' or data[index,0]=='entertainment':
            exerciseFile.write("{:s}>{:s}>{:s} \n".format("0",data[index,0],data[index,1]))
            count0+=1

print("class 1 size:{:d}".format(count1))
print("class 2 size:{:d}".format(count0))
print("Total number of samples:{:d}".format(count1+count0))
print("Number of news:{:d}".format(numberOfNews))

dataset = pd.read_csv("exercise_data.csv",sep='>')
#[print("***",val,"\n") for val in dataset.iloc[:,2].values if "xavier" in val]



        