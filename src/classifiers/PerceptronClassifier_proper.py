'''
Created on Jan 19, 2013

@author: Arvind
'''
from collections import OrderedDict
from preprocessors.Tokenizer import WordTokenizer
import fileinput
import numpy as np
import random
import subprocess
import sys
import time

#######CONSTANTS########
POSITIVE_POLARITY = 1
NEGATIVE_POLARITY = -1
OBJECTIVE_POLARITY = 0
POSITIVE_POLARITY_FOR_SCORER = "POS"
NEGATIVE_POLARITY_FOR_SCORER = "NEG"
OBJECTIVE_POLARITY_FOR_SCORER = "OBJ"
GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAIN_KEY_FILE = "../../train.key"
DEVELOPMENT_KEY_FILE = "../../dev.key"
TRAINING_DATA_DIRECTORY = "train"
DEVELOPMENT_DATA_DIRECTORY = "dev"
BASE_URL = "../../"
MAX_ITERATIONS = 30
NUMBER_OF_CLASSES = 3

quickUtilMapping = {}
quickUtilMapping[POSITIVE_POLARITY] = POSITIVE_POLARITY_FOR_SCORER
quickUtilMapping[NEGATIVE_POLARITY] = NEGATIVE_POLARITY_FOR_SCORER
quickUtilMapping[OBJECTIVE_POLARITY] = OBJECTIVE_POLARITY_FOR_SCORER

###END OF CONSTANTA########

##########Util Functions####################
'''
Get the vector representation for each file (as a list), in a
file->bag of words type data structure 
given the total distinct number of token in the vocabulary

    Each vector in the list returned will be of size |distinct_tokens_in_vocabulary|
    vector[i] = 1, if the ith token in vocabulary_list occurs in the bag
'''

def get_vector_list(words, distinct_tokens_in_vocabulary, vocabulary_list):
    training_input_vector_list = []
    for key, token_list in words.iteritems():
        document_vector = []
        document_vector = np.array([0]*len(distinct_tokens_in_vocabulary))
        for word in token_list:
            if word in vocabulary_list:
                document_vector[vocabulary_list.index(word)] += 1 
        #time.sleep(10)
        training_input_vector_list.append(document_vector)
    return training_input_vector_list

##############End Util Functions#############

#Step 1: Get the distinct tokens in the training data and the total number of tokens a.k.a bag-of-words
wordTokenizer = WordTokenizer()
words_in_training = wordTokenizer.tokenizeDirectoryAsDictionary(TRAINING_DATA_DIRECTORY)
words = wordTokenizer.tokenizeDirectory(TRAINING_DATA_DIRECTORY)
distinct_tokens = wordTokenizer.distinctTokens(words)

distinct_tokens = OrderedDict(sorted(distinct_tokens.items(), key = lambda t:t[0]))
keys_list = distinct_tokens.keys()

print "Constructing the input vector list for all the training documents"
training_input_vector_list = get_vector_list(words_in_training, distinct_tokens, keys_list)

desired_output = []
print "Finding out the desired output for each training document"
for line in fileinput.input([TRAIN_KEY_FILE]):
    temp_line = line.split(" ")
    polarity = temp_line[1].strip()
    output = POSITIVE_POLARITY if polarity == POSITIVE_POLARITY_FOR_SCORER else NEGATIVE_POLARITY if polarity == NEGATIVE_POLARITY_FOR_SCORER else OBJECTIVE_POLARITY
    desired_output.append(output)
 
print "Assigning some small random weights"   
weights_positive = []
weights_negative = []
weights_objective = []
for i in range(len(distinct_tokens)):
    weights_positive.append(random.uniform(0,0.01))
    weights_negative.append(random.uniform(0,0.01))
    weights_objective.append(random.uniform(0,0.01))

weights_dictionary = {}
weights_dictionary[POSITIVE_POLARITY] = weights_positive
weights_dictionary[NEGATIVE_POLARITY] = weights_negative
weights_dictionary[OBJECTIVE_POLARITY] = weights_objective    

# This will have a list of weights at each iteration --> Use this to find the accuracy of the development data
weights_list_positive = []
weights_list_negative = []
weights_list_objective = []
# D = [(training_input_vector_list, desired_output)]
# W = [w1, w2, ... wn]
fileName = GENERATED_FILES_DIRECTORY+"training_perceptron_data_akj"
plotFileName = GENERATED_FILES_DIRECTORY+"training_perceptron_response_akj"
f1 = open(fileName, "w")
for i in range(MAX_ITERATIONS):
    f = open(plotFileName, "w")
    print "Iteration Number: "+ str(i+1)
    document_number = 0
    accuracy = 0
    #files_remaining = len(training_input_vector_list)
    for training_document_vector in training_input_vector_list:
        actual_class = desired_output[document_number]
        document_number +=1
        #feature_function_document = feature_function(training_document_vector, actual_class)
        #feature_dictionary = {}
        
        #feature_function_document_assuming_positive = feature_function(training_document_vector, POSITIVE_POLARITY)
        #feature_function_document_assuming_negative = feature_function(training_document_vector, NEGATIVE_POLARITY)
        #feature_function_document_assuming_objective = feature_function(training_document_vector, OBJECTIVE_POLARITY)
        
        #feature_dictionary[POSITIVE_POLARITY] = feature_function_document_assuming_positive
        #feature_dictionary[NEGATIVE_POLARITY] = feature_function_document_assuming_negative
        #feature_dictionary[OBJECTIVE_POLARITY] = feature_function_document_assuming_objective
        
        #print "Finding inner product"
        p = np.dot(training_document_vector, weights_dictionary[POSITIVE_POLARITY]) 
        n = np.dot(training_document_vector, weights_dictionary[NEGATIVE_POLARITY])
        o = np.dot(training_document_vector, weights_dictionary[OBJECTIVE_POLARITY])
        #print "Done inner product"
        '''
        p,n,o = 0,0,0
        for j in range(len(weights)):
            if weights[j]!=0:
                if feature_function_document_assuming_positive[j] != 0:
                    p+=weights[j]*feature_function_document_assuming_positive[j]
                if feature_function_document_assuming_negative[j]!=0:
                    n+=weights[j]*feature_function_document_assuming_negative[j]
                if feature_function_document_assuming_objective[j]!=0:
                    n+=weights[j]*feature_function_document_assuming_objective[j]
        '''
        if p > n:
            if p > o:   #y_cap is positive class
                predicted_class = POSITIVE_POLARITY
            else:   #y_cap is objective class
                predicted_class = OBJECTIVE_POLARITY
        else:
            if n > o: #y_cap is negative class
                predicted_class = NEGATIVE_POLARITY
            else:   #y_cap is objective class
                predicted_class = OBJECTIVE_POLARITY
        if predicted_class == actual_class:
            accuracy += 1
        else:
                weights_dictionary[predicted_class] = np.subtract(weights_dictionary[predicted_class], training_document_vector)
                weights_dictionary[actual_class] = np.add(weights_dictionary[actual_class], training_document_vector)
        #files_remaining -=1
        #print "Actual Class = "+str(actual_class)
        #print "Predicted Class = "+str(predicted_class)
        f.write(quickUtilMapping[predicted_class]+"\n")
        #print "Files remaining in this iteration: "+ str(files_remaining)
    f.close()
    accuracy_rate = float(accuracy)/len(training_input_vector_list)
    line_to_write = "Accuracy on training after iteration: "+ str(i+1)+" is: "+ str(accuracy_rate)
    print line_to_write
    f1.write(line_to_write+"\n")
    weights_list_positive.append(weights_dictionary[POSITIVE_POLARITY])        #Appended the weights at the end of every pass --> each weights_list[i] will be applied on the dev data
    weights_list_negative.append(weights_dictionary[NEGATIVE_POLARITY])        #Appended the weights at the end of every pass --> each weights_list[i] will be applied on the dev data
    weights_list_objective.append(weights_dictionary[OBJECTIVE_POLARITY])        #Appended the weights at the end of every pass --> each weights_list[i] will be applied on the dev data

f1.write("\n END OF TRAINING \n");
f1.close()

'''
Prediction phase -- Just predict based on existing weights
'''

desired_output_development = []
print "Finding out the desired output for each development document"
for line in fileinput.input([DEVELOPMENT_KEY_FILE]):
    temp_line = line.split(" ")
    polarity = temp_line[1].strip()
    desired_output_development.append(polarity)

#Now weights vector needs to be used on the development data
words_in_development = wordTokenizer.tokenizeDirectoryAsDictionary(DEVELOPMENT_DATA_DIRECTORY)
print "Constructing the input vector list for all the development documents"
development_input_vector_list = get_vector_list(words_in_development, distinct_tokens, keys_list)
responseFileData = GENERATED_FILES_DIRECTORY+"development_perceptron_data_akj"
f1 = open(responseFileData, "w")

for i in range(MAX_ITERATIONS):
    print "Iteration Number: "+str(i+1)
    accuracy = 0
    #Perceptron response
    myweights_positive = weights_list_positive[i]
    myweights_negative = weights_list_negative[i]
    myweights_objective = weights_list_objective[i]
    responseFile = GENERATED_FILES_DIRECTORY+"perceptron_response_akj"
    f = open(responseFile, "w")
    dev_document_number = 0
    for development_document_vector in development_input_vector_list:
        actual_class = desired_output_development[dev_document_number]
        dev_document_number +=1
        #feature_function_document_assuming_positive = feature_function(development_document_vector, POSITIVE_POLARITY)
        #feature_function_document_assuming_negative = feature_function(development_document_vector, NEGATIVE_POLARITY)
        #feature_function_document_assuming_objective = feature_function(development_document_vector, OBJECTIVE_POLARITY)
        
        p = np.dot(development_document_vector, myweights_positive) 
        n = np.dot(development_document_vector, myweights_negative)
        o = np.dot(development_document_vector, myweights_objective)
    
        if p > n:
            if p > o:
                predicted_class = POSITIVE_POLARITY_FOR_SCORER
            else:
                predicted_class = OBJECTIVE_POLARITY_FOR_SCORER
        else:
            if n > o:
                predicted_class = NEGATIVE_POLARITY_FOR_SCORER
            else:
                predicted_class = OBJECTIVE_POLARITY_FOR_SCORER
        #print "Document predicted as: "+ predicted_class
        #print "Actual Class is: " + actual_class
        f.write(predicted_class+"\n")
        if predicted_class == actual_class:
            accuracy +=1
    f.close()
    accuracy_rate = float(accuracy)/len(development_input_vector_list)
    line_to_write = "Accuracy on development after iteration: "+ str(i+1)+" is: "+ str(accuracy_rate)
    print line_to_write
    f1.write(line_to_write+"\n")

f1.write("\n END OF DEVELOPMENT \n");
f1.close()
subprocess.call([sys.executable, '../../scorer.py', '../../train.key', plotFileName])
subprocess.call([sys.executable, '../../scorer.py', '../../dev.key', responseFile])