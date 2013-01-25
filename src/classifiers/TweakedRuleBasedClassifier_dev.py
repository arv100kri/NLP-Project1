'''
Created on Jan 18, 2013

@author: Arvind
Apply the value of the tuning parameter from the train data to the dev data
This may need to implemented as an automated script
'''
from collections import OrderedDict
from preprocessors.Tokenizer import SentimentParser, WordTokenizer
import random
import sys
import subprocess

#######Constants###########
POSITIVE_POLARITY = "positive"
NEGATIVE_POLARITY = "negative"
OBJECTIVE_POLARITY = "neutral"
POSITIVE_POLARITY_FOR_SCORER = "POS"
NEGATIVE_POLARITY_FOR_SCORER = "NEG"
OBJECTIVE_POLARITY_FOR_SCORER = "OBJ"
GENERATED_FILES_DIRECTORY = "../../generated_files/"
tuning_parameter = 6
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
                negative+=1
    abstotal = abs(positive - negative)
    if abstotal <= tuning_parameter:
        fileClassDictionary[key] = OBJECTIVE_POLARITY_FOR_SCORER
    else:
        if positive > negative: 
            fileClassDictionary[key] = POSITIVE_POLARITY_FOR_SCORER
        else:
            fileClassDictionary[key] = NEGATIVE_POLARITY_FOR_SCORER

fileName = GENERATED_FILES_DIRECTORY+"tweaked_response_dev"+str(tuning_parameter)
f = open(fileName, "w")

fileClassDictionary = OrderedDict(sorted(fileClassDictionary.items(), key=lambda t: t[0]))
for key, value in fileClassDictionary.iteritems():
    print key+"\t This document is of polarity: "+value
    print "Its text is:-->" + str(fileDictionary[key])
    f.write(value+"\n")
f.close()

subprocess.call([sys.executable, '../../scorer.py', '../../dev.key', fileName])