# -*- coding: utf-8 -*-


import pandas as pd
from collections import defaultdict
from collections import OrderedDict


def cleanSplitText(text):
    filterCharacters = "_\'*;!%&$#()-0123456789.,/?:'\""
    cleanText = ''.join(list(filter(lambda ch: ch not in filterCharacters,text)))
    cleanText = cleanText.split(' ')
    return cleanText
    
# Importing the dataset
dataset = pd.read_csv('exercise_data.csv',sep='>')
X_raw = dataset.iloc[:, 2].values.tolist()
y = dataset.iloc[:, 0].values.tolist()

wordBag = defaultdict(lambda : 0 )
for i in range(len(y)):
    text = cleanSplitText(X_raw[i])
    
    for word in text:
        #Filter words of size 1 or 0
        if len(word)>1:
            if len(word)>3 and word[-1]=='s':
                word=word[0:-1]
            wordBag[word] += 1

filteredWords = list(filter(lambda x:x[1]>=1,wordBag.items()))
numOfWords = len(filteredWords)
print("Number of different words:{:d}".format(numOfWords))    
finalWordBag = OrderedDict(sorted(filteredWords, key=lambda word: word[1],reverse=True))
finalWordBag = OrderedDict(sorted(list(finalWordBag.items())[:], key=lambda word: word[0],reverse=True))

with open("bow.txt","w") as bowFile:
    count = 0
    for k,v  in finalWordBag.items():
        #print(count,",",k,":",v)
        bowFile.write("{:s},{:d}\n".format(k,count))
        count+=1
    
