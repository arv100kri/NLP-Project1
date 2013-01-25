'''
Created on Jan 22, 2013

@author: Arvind
'''

import matplotlib.pyplot as plt
import subprocess
import sys

GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAINING_KEY = '../../train.key'
DEVELOPMENT_KEY = '../../dev.key'
SCORER_SCRIPT = '../../scorer.py'
TRAINING_FILES_PATTERN = GENERATED_FILES_DIRECTORY+"smoothing_response_"

X_values = []
for i in range(0,8):
    X_values.append(i)

#Got the values by manually reading the output from the scorer script
print X_values
Y_values1 = [78.71, 99.51, 99.76, 99.76, 99.88, 99.88, 100.00, 100.00]    #Training

Y_values2 = [67.64, 76.36, 77.09, 77.09, 73.45, 73.09, 72.00, 71.27]    #Development
#plt.ticklabel_format(style='sci', axis='x', scilimits=(-8, 1))
plt.xlim(0, 8)
#plt.xscale(0.1)
plt.xlabel('Value of ALPHA (10^-x)')
plt.ylabel("Accuracy")
plt.plot(X_values, Y_values1, color = 'r', marker = 'o', label='Training')
plt.plot(X_values, Y_values2, color = 'g', marker = 'o', label='Development')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
plt.show()