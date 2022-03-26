import random

import numpy as np

from sigmoid import sigmoid, sigmoid_prime
from data import get_test_data, get_training_data


class Network(object):
    def __init__(self, layer_sizes):
        self.biases = [np.random.randn(x) for x in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data_length = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, training_data_length, mini_batch_size)
            ]

            for batch in mini_batches:
                self.update_with_mini_batch(batch, eta)

            print("--------------")
            print(f"Epoch {i} complete")

            accuracy = None
            if test_data:
                accuracy = self.evaluate_epoch(test_data)

            print(f"accuracy: {accuracy} / {len(test_data)}")
        


    def update_with_mini_batch(self, batch, eta):
        total_delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        total_delta_w = [np.zeros(np.shape(w)) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y) 
            total_delta_b = [tdb + db for tdb, db in zip(total_delta_b, delta_b)]
            total_delta_w = [tdw + dw for tdw, dw in zip(total_delta_w, delta_w)]
        # print(f"Old Biases: {self.biases}")
        self.biases = [b - eta * (tdb / len(batch)) for b, tdb in zip(self.biases, total_delta_b)]
        
        # print(f"New Biases: {self.biases}")
        self.weights = [w - eta * (tdw / len(batch)) for w, tdw in zip(self.weights, total_delta_w)]
        

    def backprop(self, x, y):
        delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        delta_w = [np.zeros(np.shape(w)) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
           
            layer_z = np.add(np.dot(w, activation), b)
            zs.append(layer_z)

            activation = sigmoid(layer_z)
            
            activations.append(activation)

        error = sigmoid_prime(zs[-1]) * self.cost_derivative(activations[-1], y)
        # print(f"sigmoid prime z: {sigmoid_prime(zs[-1])[0]}")
        # print(f"last layer error: {error}")
        delta_b[-1] = error
        delta_w[-1] = [e * activations[-2] for e in error]
        for l in range(2, self.layer_count):
            z = zs[-l]
            w =  np.transpose(self.weights[-l + 1])
            # print(f"weights: {self.weights[-l + 1]}")
            # print(f"transposed weights: {w}")
            # print(f"next error: {error}")
            # print(f"dot error and weights: {np.dot(w, error)})")
            error = np.dot(w, error) * sigmoid_prime(z)
            # print(f"layer error: {error}")
            delta_b[-l] = error
            delta_w[-l] = [e * activations[-l-1] for e in error]

        # print(f"Bias Delta: {delta_b}")
        # print(f"Weight Delta: { delta_w }")
        return (delta_b, delta_w)



    def cost_derivative(self, a, y) -> np.array:
        return 2 * (a - y)

    def evaluate_epoch(self, test_data):
        correct_count = 0
        for x, y in test_data:
            activation = x
            for b, w in zip(self.biases, self.weights):
                layer_z = np.add(np.dot(w, activation), b)
                activation = sigmoid(layer_z)
            
            maxIndex = np.argmax(activation)

            if y[maxIndex] == 1:
                correct_count += 1

        return correct_count
        


