
'''
Created on Jan 22, 2013

@author: Arvind
'''
from scipy.spatial.distance import correlation
import fileinput
import matplotlib.pyplot as plt
import time

GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAINING_KEY = '../../train.key'
DEVELOPMENT_KEY = '../../dev.key'
SCORER_SCRIPT = '../../scorer.py'
WEIGHTS_FILE = "averaged_weight_vector_akj"
LOG_POSITIVE_FILE = "versus_file_bayes"
DISTINCT_TOKENS = 11083


for line in fileinput.input([GENERATED_FILES_DIRECTORY+WEIGHTS_FILE]):
    if len(line) == 0:
        break
    lister = line.split(" ")
    positive_weights = []
    for weight in lister[:DISTINCT_TOKENS]:
        append_weight = float(weight)
        positive_weights.append(append_weight)
#Need to plot positive weights against the values in versus_bayes
log_weights = []
actual_positives = []
for line in fileinput.input([GENERATED_FILES_DIRECTORY+LOG_POSITIVE_FILE]):
    lister = line.split("\t")
    log_weights.append(float(lister[1]))
    actual_positives.append(positive_weights[int(lister[2])])

plt.plot(log_weights, actual_positives, marker ='o', ls ='')
plt.xlabel("Log weights of positive tokens (Naive Bayes Classifier - ALPHA = 10^-3)")
plt.ylabel("Weights of the positive tokens (Perceptron Classifier)")
print "Corelation is: "+ str(correlation(log_weights, actual_positives))

#plt.plot(log_weights, 'rs')
plt.show()
