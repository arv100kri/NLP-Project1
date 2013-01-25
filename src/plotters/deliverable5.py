'''
Created on Jan 22, 2013

@author: Arvind
'''

import fileinput
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import correlation


GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAINING_KEY = '../../train.key'
DEVELOPMENT_KEY = '../../dev.key'
SCORER_SCRIPT = '../../scorer.py'
TRAINING_FILES_PATTERN = GENERATED_FILES_DIRECTORY+"smoothing_response_MIN_MAX"
POSITIVE_POLARITY_FOR_SCORER = "POS"
NEGATIVE_POLARITY_FOR_SCORER = "NEG"
OBJECTIVE_POLARITY_FOR_SCORER = "OBJ"

minValue = "-7"
maxValue = "0"

def plotter(Value):
    X_pos = []
    X_neg = []
    X_obj = []
    
    fileName = TRAINING_FILES_PATTERN+Value
    for line in fileinput.input([fileName]):
        split = line.split("\t")
        if split[0] == POSITIVE_POLARITY_FOR_SCORER:
            X_pos.append(float(split[1]))
        elif split[0] == NEGATIVE_POLARITY_FOR_SCORER:
            X_neg.append(float(split[1]))
        else:
            X_obj.append(float(split[1]))
    return X_pos, X_neg, X_obj


max_pos, max_neg, max_obj = plotter(maxValue)
min_pos, min_neg, min_obj = plotter(minValue)

max = max_pos + max_neg + max_obj
min = min_pos + min_neg + min_obj

print max
print min

print "Corelation is: "+ str(correlation(max, min))

plt.plot(max, min, marker = 'o', ls='')
plt.xlabel("Value of alpha as 10^0")
plt.ylabel("Value of alpha as 10^-7")
plt.show()  