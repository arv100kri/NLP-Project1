'''
Created on Jan 18, 2013

@author: Arvind
'''
from collections import defaultdict, OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize
import fileinput
import os
import random

class WordTokenizer(object):
    '''
    Pre-processing step 1
    1. Tokenize all the files in the train data
    2. Downcase the tokens
    '''
    __BASE_URL = "../../"

    def __init__(self):
        '''
        Constructor
        '''
        
    def tokenizeString(self, mystr):
        sentences = sent_tokenize(mystr)
        word_tokens = []
        for t in sentences:
            words = word_tokenize(t)
            for word in words:
                if self.isSanitized(word):
                    #Downcase the word
                    word_tokens.append(word.lower())    
        return word_tokens
    
    def tokenizeFile(self, fileName):
        f = open(fileName, "r")
        text = f.read()
        f.close()
        return self.tokenizeString(text)
    
    def getBaseUrl(self):
        return self.__BASE_URL
    
    def tokenizeDirectory(self, directoryName):
        words = []
        for fileName in os.listdir(self.getBaseUrl()+directoryName):
            fileName = self.getBaseUrl()+directoryName+"/"+fileName
            print "looking in "+ fileName
            for word in self.tokenizeFile(fileName): 
                words.append(word)
        return words
    
    def tokenizeDirectoryAsDictionary(self, directoryName):
        dictionaryFileName = {}
        for fileName in os.listdir(self.getBaseUrl()+directoryName):
            fileName = self.getBaseUrl()+directoryName+"/"+fileName
            print "looking in "+ fileName
            dictionaryFileName[fileName] = self.tokenizeFile(fileName)
        return dictionaryFileName
            
            
    #The value will be a bag of words in addition to a list of words
    #fileName -> (list, bag)
    def tokenizeDirectoryAsDictionary_special(self, directoryName):
        dictionaryFileName = {}
        for fileName in os.listdir(self.getBaseUrl()+directoryName):
            fileName = self.getBaseUrl()+directoryName+"/"+fileName
            print "looking in "+ fileName
            list_of_tokens_for_fileName = self.tokenizeFile(fileName)
            dictionaryFileName[fileName] = (list_of_tokens_for_fileName, self.distinctTokens(list_of_tokens_for_fileName))
        return dictionaryFileName
        
    
    ''''
        Condition for sanity
            --> Must be all alphabets
    '''
    def isSanitized(self, word):
        return word.isalpha()
    
    def distinctTokens(self, words):
        d = defaultdict(int)
        for word in words:
            d[word]+=1
        return d

class SentimentParser(object):
    __FILE_LOCATION = "../../sentiment-vocab.tff"
    __WORD_POSITION = "word1"
    __PRIOR_POLARITY = "priorpolarity"
    __OBJECTIVE_POLARITY = "neutral"
    def __init__(self):
        '''
        Constructor
        '''
    
    '''
        This function should be called at the start of execution
        Returns a dictionary mapping each word to its sentiment
        s = [('word1': 'POS'), ('word2': 'NEG') , ('word3': 'OBJ')]
        where word1, word2, word3 etc., are from the sentiment file
    '''
    def wordsClassification(self):
        classification = defaultdict(list)
        linelist = []
        for content in fileinput.input([self.__FILE_LOCATION]):
            #print content
            linelist.append(content)
            line_split = content.split(" ")
            if len(line_split) > 0:
                index1 = -1
                index2 = -1
                for words in line_split:
                    if self.__WORD_POSITION in words:   #Found a match like word1=
                        index1 = line_split.index(words);
                    elif self.__PRIOR_POLARITY in words:    #Found a match like priorpolarity=
                        index2 = line_split.index(words);
                if index1>=0 and index2>=0:
                    word = line_split[index1].split("=")[1]
                    sense = line_split[index2].split("=")[1].strip()
                    '''
                    if word in classification:
                        print word
                        print content.strip()
                        print classification[word]
                        
                        for w in line_split:
                            if self.__PRIOR_POLARITY in w:
                                i = line_split.index(w)
                                break;
                        sense1 = line_split[i].split("=")[1].strip()
                        #sense2 = classification[word]
                        #print sense1+" "+sense2
                        #if sense1!=sense2:
                        #Add the new sense
                    else:
                        '''
                    classification[word].append(sense);
        return OrderedDict(sorted(classification.items(), key=lambda t: t[0]))
    
    '''
        Lookup dictionary must be of the form
        d[word]=>['sense1', 'sense2',...]
        If no sense is found, return 'objective' by default
        If found, return a randome sense from sense1, sense2 etc.,    
    '''
    def getSenseOfWord(self, word, lookup_dictionary):
        if word in lookup_dictionary:
            return lookup_dictionary[word]
        else:
            return self.__OBJECTIVE_POLARITY