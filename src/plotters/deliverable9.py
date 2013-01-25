'''
Created on Jan 24, 2013

@author: Arvind
'''
import fileinput
import matplotlib.pyplot as plt

GENERATED_FILES_DIRECTORY = "../../generated_files/"
TRAINING_KEY = '../../train.key'
DEVELOPMENT_KEY = '../../dev.key'
SCORER_SCRIPT = '../../scorer.py'
DATA_FILE_TRAINING = "training_perceptron_data_akj"
DATA_FILE_DEVELOPMENT = "averaged_development_perceptron_data_akj"
MAX_ITERATIONS = 30
iterations = []
training_accuracy = []
development_accuracy = []
count = 0
for line in fileinput.input([GENERATED_FILES_DIRECTORY+DATA_FILE_TRAINING]):
    if count == MAX_ITERATIONS:
        break
    count += 1
    iterations.append(count)
    splitter = line.split(": ")
    accuracy = float(splitter[2])*100
    training_accuracy.append(accuracy)
fileinput.close()

count = 0
for line in fileinput.input([GENERATED_FILES_DIRECTORY+DATA_FILE_DEVELOPMENT]):
    if count == MAX_ITERATIONS:
        break
    count += 1
    splitter = line.split(": ")
    accuracy = float(splitter[2])*100
    development_accuracy.append(accuracy)
fileinput.close() 
    
plt.plot(iterations, training_accuracy, color = 'r', label='Training')
plt.plot(iterations, development_accuracy, color = 'g', label = 'Development')
plt.xlabel('Number of iterations')
plt.ylabel("Accuracy (%)")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
plt.show()
