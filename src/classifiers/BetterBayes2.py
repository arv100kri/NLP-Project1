'''
Created on Jan 19, 2013

@author: Arvind
Transformed weight normlized Bayes
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
POS_INFINITY = 99999
POW = 3 #Other values are 0,1,2,3 (best),4,5,6
ALPHA = math.pow(10, POW*-1)

word_file_occurrence_dictionary_positive = defaultdict(int)  #--> word-> number_of_files_which_have_word
word_file_occurrence_count_dictionary_positive = defaultdict(list) #--> word --> list of int, representing number of occurrence in each file
                                            #list is empy if word_file_occurrence_dictionary[word] = 0

word_file_occurrence_dictionary_negative = defaultdict(int)  #--> word-> number_of_files_which_have_word
word_file_occurrence_count_dictionary_negative = defaultdict(list) #--> word --> list of int, representing number of occurrence in each file
                                            #list is empy if word_file_occurrence_dictionary[word] = 0

word_file_occurrence_dictionary_objective = defaultdict(int)  #--> word-> number_of_files_which_have_word
word_file_occurrence_count_dictionary_objective = defaultdict(list) #--> word --> list of int, representing number of occurrence in each file
                                            #list is empy if word_file_occurrence_dictionary[word] = 0

####END OF CONSTANTS#######

########Util Functions#############
def twcnb(word, except_class_file_bag, vocabulary_size, sentiment):
    thetha_word_class = 0.0
    for bag in except_class_file_bag.values():
        if word in bag:   
            d_i_j = math.log(bag[word]+1)
        else:
            d_i_j = 0
        num = len(except_class_file_bag)
        
        if sentiment == POSITIVE_POLARITY_FOR_SCORER:
            if word not in word_file_occurrence_dictionary_positive:
                return POS_INFINITY
            
            den = word_file_occurrence_dictionary_positive[word]
            den2 = sum(word_file_occurrence_count_dictionary_positive[word])
            
            den1 = math.sqrt(float(sum(i*2 for i in word_file_occurrence_count_dictionary_positive[word])))
            d_i_j *= num/den
            d_i_j /= den1
            thetha_word_class += (d_i_j + ALPHA)/(den2 + ALPHA * vocabulary_size)
        
        elif sentiment == NEGATIVE_POLARITY_FOR_SCORER:
            if word not in word_file_occurrence_dictionary_negative:
                return POS_INFINITY
            
            den = word_file_occurrence_dictionary_negative[word]
            den2 = sum(word_file_occurrence_count_dictionary_negative[word])
            
            den1 = math.sqrt(float(sum(i*2 for i in word_file_occurrence_count_dictionary_negative[word])))
            d_i_j *= num/den
            d_i_j /= den1
            thetha_word_class += (d_i_j + ALPHA)/(den2 + ALPHA * vocabulary_size)
        else:
            if word not in word_file_occurrence_dictionary_objective:
                return POS_INFINITY
            
            den = word_file_occurrence_dictionary_objective[word]
            den2 = sum(word_file_occurrence_count_dictionary_objective[word])
            
            den1 = math.sqrt(float(sum(i*2 for i in word_file_occurrence_count_dictionary_objective[word])))
            d_i_j *= num/den
            d_i_j /= den1
            thetha_word_class += (d_i_j + ALPHA)/(den2 + ALPHA * vocabulary_size)
    
    return math.log(thetha_word_class)

def total(bag):
    total = 0
    #return the total count of words in a bag
    for value in bag.values():
        total += value
    return total
'''
#files_word_bag: file --> bag_of_words dictionary
#No longer needed --> replaced with a lookup
def files_having_word(files_words_bag, word):
    count = 0
    length_square = 0.0
    length = 0
    for this_bag in files_words_bag.values():
        if word in this_bag:
            count+=1
            length_square += math.pow(this_bag[word], 2)
            length += this_bag[word]
    return count, float(math.sqrt(length_square)), length 
'''
####################################

#Step 1: Get the distinct tokens in the training data and the total number of tokens a.k.a bag-of-words
wordTokenizer = WordTokenizer()
words_in_training = wordTokenizer.tokenizeDirectoryAsDictionary(TRAINING_DATA_DIRECTORY)
training_document_bag_dictionary = {}
distinct_tokens = defaultdict(int)
for key, token_list in words_in_training.iteritems():
    bag_of_words = defaultdict(int)
    for word in token_list:
        bag_of_words[word]+=1
        distinct_tokens[word]+=1
    training_document_bag_dictionary[key] = bag_of_words

training_document_bag_dictionary = OrderedDict(sorted(training_document_bag_dictionary.items(), key=lambda t: t[0]))

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

total_count = total_positive + total_negative + total_objective

#------------------Improvement over Naive Bayes, requires calculation of probability of a word
#------------------in all classes except c. This data structure helps in that
#negative+objective
except_positive_training_document_bag_dictionary = dict(negative_training_document_bag_dictionary.items() + objective_training_document_bag_dictionary.items())
#positive+objective 
except_negative_training_document_bag_dictionary = dict(positive_training_document_bag_dictionary.items() + objective_training_document_bag_dictionary.items())
#positive+negative
except_objective_training_document_bag_dictionary = dict(positive_training_document_bag_dictionary.items() + negative_training_document_bag_dictionary.items())


probability_positive = float(training_total_positive_documents)/ training_total_documents
probability_negative = float(training_total_negative_documents)/ training_total_documents
probability_objective = float(training_total_objective_documents)/ training_total_documents

log_positive = math.log(probability_positive)
log_negative = math.log(probability_negative)
log_objective = math.log(probability_objective)

words_in_development = wordTokenizer.tokenizeDirectoryAsDictionary_special(DEVELOPMENT_DATA_DIRECTORY)
words_in_development = OrderedDict(sorted(words_in_development.items(), key=lambda t: t[0]))


#-----------Building the word_file_dictionary and word_file_occurrence_dictionary for convenience/speedup-----------
print "Building convenience lists...."
counter_key = len(distinct_tokens)
for key in distinct_tokens:
    for fileName, bag_of_words in positive_training_document_bag_dictionary.iteritems():
        if key in bag_of_words:
            word_file_occurrence_dictionary_positive[key]+=1
            word_file_occurrence_count_dictionary_positive[key].append(bag_of_words[key])
    for fileName, bag_of_words in negative_training_document_bag_dictionary.iteritems():
        if key in bag_of_words:
            word_file_occurrence_dictionary_negative[key]+=1
            word_file_occurrence_count_dictionary_negative[key].append(bag_of_words[key])
    for fileName, bag_of_words in objective_training_document_bag_dictionary.iteritems():
        if key in bag_of_words:
            word_file_occurrence_dictionary_objective[key]+=1
            word_file_occurrence_count_dictionary_objective[key].append(bag_of_words[key]) 
    counter_key -=1
    print "Another "+str(counter_key)+" more to go"
    
print "Things should be faster now..."
responseFile = GENERATED_FILES_DIRECTORY+"better2_naive_response_"+str(POW)
f = open(responseFile, "w")
for key, token_list in words_in_development.iteritems():
    sentiment = ""
    p = 0
    n = 0
    o = 0
    #token_list[0] --> List of words in the file
    #token_list[1] --> bag of words of the file
    for word in token_list[0]:
        #Step 2.1
        #Ti for the word
        count_word = token_list[1][word]
        w_positive_word = twcnb(word, except_positive_training_document_bag_dictionary, total_count, POSITIVE_POLARITY_FOR_SCORER)
        
        w_negative_word = twcnb(word, except_negative_training_document_bag_dictionary, total_count, NEGATIVE_POLARITY_FOR_SCORER)
        
        w_objective_word = twcnb(word, except_objective_training_document_bag_dictionary, total_count, OBJECTIVE_POLARITY_FOR_SCORER)
        
        p += count_word * w_positive_word
        n += count_word * w_negative_word
        o += count_word * w_objective_word
    #Argmin of the values
    if p < n:
        if p < o:
            sentiment = POSITIVE_POLARITY_FOR_SCORER
        else:
            sentiment = OBJECTIVE_POLARITY_FOR_SCORER
    else:
        if n < o:
            sentiment = NEGATIVE_POLARITY_FOR_SCORER
        else:
            sentiment = OBJECTIVE_POLARITY_FOR_SCORER
            
    print "Document Name: "+key+" is classified as: "+sentiment
    f.write(sentiment+"\n")
f.close()
subprocess.call([sys.executable, '../../scorer.py', '../../dev.key', responseFile])