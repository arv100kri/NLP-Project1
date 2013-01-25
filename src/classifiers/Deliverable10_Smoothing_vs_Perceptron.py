'''
Created on Jan 19, 2013

@author: Arvind
'''
from collections import OrderedDict, defaultdict
from preprocessors.Tokenizer import WordTokenizer
import fileinput
import math
import subprocess
import sys

#######CONSTANTS########
POSITIVE_POLARITY = "positive"
NEGATIVE_POLARITY = "negative"
OBJECTIVE_POLARITY = "neutral"
POSITIVE_POLARITY_FOR_SCORER = "POS"
NEGATIVE_POLARITY_FOR_SCORER = "NEG"
OBJECTIVE_POLARITY_FOR_SCORER = "OBJ"
GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAIN_KEY_FILE = "../../train.key"
TRAINING_DATA_DIRECTORY = "train"
DEVELOPMENT_DATA_DIRECTORY = "dev"
NEG_INFINITY = -99999
POW = 3 #Other values are 0,1,2,3 (best),4,5,6
ALPHA = math.pow(10, POW*-1)
####END OF CONSTANTS#######

########Util Functions#############
def probabilityOfWordInAFile(word, file_bag):   #Probably not necessary. Keep it anyway
    if word not in file_bag:
        return 0
    count = file_bag[word]
    total = 0
    for value in file_bag.values():
        total+=value
    return float(count)/total

def probabilityOfWordInSeveralDocumentsPutTogether(word, document_bag_dictionary, vocabulary_size):
    count = 0
    total = 0
    for bag in document_bag_dictionary.values():
        for value in bag.values():
            total += value
        if word in bag:
            count += bag[word]
    #Smoothing occurs here
    count += ALPHA
    total += vocabulary_size*ALPHA
    #End Smoothing
    return float(count)/total

def total(bag):
    total = 0
    #return the total count of words in a bag
    for value in bag.values():
        total += value
    return total;

def getdistinct_tokens(file_to_bag_mapping):
    distincts = defaultdict(int)
    for bag in file_to_bag_mapping.values():
        for key in bag:
            distincts[key]+=1
    return distincts.keys()

####################################

#Step 1: Get the distinct tokens in the training data and the total number of tokens a.k.a bag-of-words
wordTokenizer = WordTokenizer()
words_in_training = wordTokenizer.tokenizeDirectoryAsDictionary(TRAINING_DATA_DIRECTORY)
training_document_bag_dictionary = {}
for key, token_list in words_in_training.iteritems():
    bag_of_words = defaultdict(int)
    for word in token_list:
        bag_of_words[word]+=1
    training_document_bag_dictionary[key] = bag_of_words

training_document_bag_dictionary = OrderedDict(sorted(training_document_bag_dictionary.items(), key=lambda t: t[0]))

distinct_tokens = getdistinct_tokens(training_document_bag_dictionary)
print len(distinct_tokens)

#print training_document_bag_dictionary   

positive_training_document_bag_dictionary = {}
negative_training_document_bag_dictionary = {}
objective_training_document_bag_dictionary = {}

total_positive = 0
total_negative = 0
total_objective = 0

for line in fileinput.input([TRAIN_KEY_FILE]):
    temp_line = line.split(" ")
    fileName = temp_line[0]
    sense = temp_line[1].strip()
    fileName = "../../"+fileName #Make it into a suitable format
    if sense == POSITIVE_POLARITY_FOR_SCORER:   #document is a positive one 
        #Add the fileName -> bag-of-words(fileName) to a dict
        positive_training_document_bag_dictionary[fileName] = training_document_bag_dictionary[fileName]
        total_positive += total(training_document_bag_dictionary[fileName])
    elif sense == NEGATIVE_POLARITY_FOR_SCORER:
        negative_training_document_bag_dictionary[fileName] = training_document_bag_dictionary[fileName]
        total_negative += total(training_document_bag_dictionary[fileName])
    elif sense == OBJECTIVE_POLARITY_FOR_SCORER:
        objective_training_document_bag_dictionary[fileName] = training_document_bag_dictionary[fileName]
        total_objective += total(training_document_bag_dictionary[fileName])
    
training_total_documents = len(training_document_bag_dictionary)
training_total_positive_documents = len(positive_training_document_bag_dictionary)
training_total_negative_documents = len(negative_training_document_bag_dictionary)
training_total_objective_documents = len(objective_training_document_bag_dictionary)

probability_positive = float(training_total_positive_documents)/ training_total_documents
probability_negative = float(training_total_negative_documents)/ training_total_documents
probability_objective = float(training_total_objective_documents)/ training_total_documents


words_in_development = wordTokenizer.tokenizeDirectoryAsDictionary(DEVELOPMENT_DATA_DIRECTORY)
words_in_development = OrderedDict(sorted(words_in_development.items(), key=lambda t: t[0]))

total_count = total_positive + total_negative + total_objective
log_word_probability_positive = {} #Mapping between word -> log(P(word|class= Positive)

log_pos = math.log(probability_positive)
log_neg = math.log(probability_negative)
log_obj = math.log(probability_objective)

responseFile = GENERATED_FILES_DIRECTORY+"comparison_smoothing_response_"+str(POW)
f = open(responseFile, "w")
for key, token_list in words_in_development.iteritems():
    sentiment = ""
    p = 0
    n = 0
    o = 0
    #Step 2: P(POS|Document) P(NEG|Document} P(OBJ|Document)
    for word in token_list:
        #Step 2.1
        #P(k| POS) P(k| NEG) P(k| OBJ) for each word in this document
        
        probability_word_positive_documents = probabilityOfWordInSeveralDocumentsPutTogether(word, positive_training_document_bag_dictionary, total_count)
        log_probability_positive = math.log(probability_word_positive_documents) if probability_word_positive_documents > 0 else NEG_INFINITY
        p+=log_probability_positive
        
        probability_word_negative_documents = probabilityOfWordInSeveralDocumentsPutTogether(word, negative_training_document_bag_dictionary, total_count)
        log_probability_negative = math.log(probability_word_negative_documents) if probability_word_negative_documents > 0 else NEG_INFINITY
        n+=log_probability_negative
        
        probability_word_objective_documents = probabilityOfWordInSeveralDocumentsPutTogether(word, objective_training_document_bag_dictionary, total_count)
        log_probability_objective = math.log(probability_word_objective_documents) if probability_word_objective_documents > 0 else NEG_INFINITY
        o+=log_probability_objective
        #Step 2.2: P(C|Document) = P(C) + all the P(k|C) --> Here in log form
        p+=log_pos
        n+=log_neg
        o+=log_obj
        if word in log_word_probability_positive:
                new_p = log_word_probability_positive[word][0]+p
                log_word_probability_positive[word] = (new_p, distinct_tokens.index(word))
        elif word in distinct_tokens:
                log_word_probability_positive[word] = (p, distinct_tokens.index(word))
            
    if p > n:
        if p > o:
            sentiment = POSITIVE_POLARITY_FOR_SCORER
        else:
            sentiment = OBJECTIVE_POLARITY_FOR_SCORER
    else:
        if n > o:
            sentiment = NEGATIVE_POLARITY_FOR_SCORER
        else:
            sentiment = OBJECTIVE_POLARITY_FOR_SCORER
    
    print "Document Name: "+key+" is classified as: "+sentiment
    f.write(sentiment+"\n")
f.close()

versus_file_bayes = GENERATED_FILES_DIRECTORY+"versus_file_bayes"
f = open(versus_file_bayes, "w")
for key, value in log_word_probability_positive.iteritems():
    f.write(key+"\t")
    f.write(str(value[0])+"\t")
    f.write(str(value[1])+"\n")
f.close()
subprocess.call([sys.executable, '../../scorer.py', '../../dev.key', responseFile])