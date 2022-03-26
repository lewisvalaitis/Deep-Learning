import random
import json
import sys

import numpy as np

from sigmoid import sigmoid, sigmoid_prime
from cost_functions import CrossEntropyCost

class Network(object):

    def __init__(self, layer_sizes, cost=CrossEntropyCost):
        self.layer_count = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(x) for x in self.layer_sizes[1:]]
        self.weights = [
            np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for batch in mini_batches:    
                self.update_with_mini_batch(batch, eta, lmbda, n)

            print(f"Epoch {i} training completed")
            if monitor_evaluation_accuracy:
                accuracy = self.evaluate_epoch(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy}")

        return evaluation_accuracy

    def update_with_mini_batch(self, batch, eta, lmbda, n):
        total_delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        total_delta_w = [np.zeros(np.shape(w)) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y) 
            
            total_delta_b = [tdb + db for tdb, db in zip(total_delta_b, delta_b)]
            total_delta_w = [tdw + dw for tdw, dw in zip(total_delta_w, delta_w)]

        self.biases = [b - eta * (tdb / len(batch)) for b, tdb in zip(self.biases, total_delta_b)]
        self.weights = [(1 - eta * (lmbda/n)) * w - (eta / len(batch)) * tdw  for w, tdw in zip(self.weights, total_delta_w)]


    def backprop(self, x, y):
        delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        delta_w = [np.zeros(np.shape(w)) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            layer_z = np.dot(w, activation) + b
            zs.append(layer_z)

            activation = sigmoid(layer_z)
            
            activations.append(activation)

        error = self.cost.delta(zs[-1], activations[-1], y)

        delta_b[-1] = error
        delta_w[-1] = [e * activations[-2] for e in error]
        for l in range(2, self.layer_count):
            z = zs[-l]
            w = np.transpose(self.weights[-l + 1])
            error = np.dot(w, error) * sigmoid_prime(z)

            delta_b[-l] = error
            delta_w[-l] = [e * activations[-l-1] for e in error]

        return (delta_b, delta_w)
    


    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate_epoch(self, test_data):
        correct_count = 0
        for x, y in test_data:
            a = self.feed_forward(x)
            
            maxIndex = np.argmax(a)
            correct_count += int(y[maxIndex] == 1)
        return correct_count / len(test_data)


    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()




def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
