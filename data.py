import random
import csv

import numpy as np


def get_shifted_data(data):

    shifted_data = []

    for input, output in data:
        random_shift = random.randint(0, 784)
        new_input = input[random_shift:]
        new_input = np.append(new_input, input[:random_shift]) 
        shifted_data.append((new_input, output))

    return shifted_data
    





def get_test_data():
    file = open('mnist_test.csv')
    csv_reader = csv.reader(file)

    test_data = []
    for row in csv_reader:
        input = np.array([float(p) / 255 for p in row[1:]])
        output = output_from_label(row[0])

        test_data.append((input, output))
    
    return test_data


def get_training_data():
    file = open('mnist_train.csv')
    csv_reader = csv.reader(file)

    training_data = []

    for row in csv_reader:
        input = np.array([float(p) / 255 for p in row[1:]])
        output = output_from_label(row[0])

        training_data.append((input, output))
    
    return training_data



def output_from_label(label: str):
    index = int(label)

    output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    output[index] = 1.0
    return output

# inputs, outputs = get_training_data()

# print(f"input: {inputs[0]}")
# print(f"output: {outputs[0]}")
get_test_data()