'''
Created on Jan 18, 2013

@author: Arvind
'''
from collections import OrderedDict
from preprocessors.Tokenizer import SentimentParser, WordTokenizer
import random
import subprocess
import sys

#######Constants###########
POSITIVE_POLARITY = "positive"
NEGATIVE_POLARITY = "negative"
OBJECTIVE_POLARITY = "neutral"
POSITIVE_POLARITY_FOR_SCORER = "POS"
NEGATIVE_POLARITY_FOR_SCORER = "NEG"
OBJECTIVE_POLARITY_FOR_SCORER = "OBJ"
GENERATED_FILES_DIRECTORY = "../../generated_files/"
#####End of Constants#####


classificationSentimentDictionary = SentimentParser().wordsClassification()
#print classificationSentimentDictionary
tokenizer = WordTokenizer()
fileDictionary = tokenizer.tokenizeDirectoryAsDictionary("dev")
fileDictionary = OrderedDict(sorted(fileDictionary.items(), key=lambda t: t[0]))
fileClassDictionary = {}
for key, value_as_list in fileDictionary.iteritems():
    positive = 0
    negative = 0
    for word in value_as_list:
        if word in classificationSentimentDictionary:
            word_sense_list = classificationSentimentDictionary[word]
            word_sense = word_sense_list[random.randint(0, len(word_sense_list)-1)]
            if word_sense == POSITIVE_POLARITY:
                positive+=1
            elif word_sense == NEGATIVE_POLARITY:
                negative-=1
    total = positive + negative
    if total >0:
        fileClassDictionary[key] = POSITIVE_POLARITY_FOR_SCORER
    else:
        fileClassDictionary[key] = NEGATIVE_POLARITY_FOR_SCORER

fileName = GENERATED_FILES_DIRECTORY+"rule_response"
f = open(fileName, "w")

fileClassDictionary = OrderedDict(sorted(fileClassDictionary.items(), key=lambda t: t[0]))
for key, value in fileClassDictionary.iteritems():
    print key+"\t This document is of polarity: "+value
    print "Its text is:-->" + str(fileDictionary[key])
    f.write(value+"\n")
    
f.close()
    
subprocess.call([sys.executable, '../../scorer.py', '../../dev.key', fileName])   