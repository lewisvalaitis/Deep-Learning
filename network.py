import random

import numpy as np

from sigmoid import sigmoid, sigmoid_prime


class Network(object):
    def __init__(self, layer_sizes):
        self.biases = [np.random.randn(x) for x in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)
        

    def SGD(self, training_data, epochs, mini_batch_size: int, eta, test_data=None):
        training_data_length = len(training_data)
        for i in range(epochs):
            print("--------------")
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, training_data_length, mini_batch_size)
            ]

            for batch in mini_batches:    
                self.update_with_mini_batch(batch, eta)

            print(f"Epoch {i} complete")

            accuracy = None

            if test_data:
                accuracy = self.evaluate_epoch(test_data)
                print(f"Current Accuracy: {accuracy} / {len(test_data)}")
        
    def update_with_mini_batch(self, batch, eta):
        total_delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        total_delta_w = [np.zeros(np.shape(w)) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y) 
            
            total_delta_b = [tdb + db for tdb, db in zip(total_delta_b, delta_b)]
            total_delta_w = [tdw + dw for tdw, dw in zip(total_delta_w, delta_w)]

        self.biases = [b - eta * (tdb / len(batch)) for b, tdb in zip(self.biases, total_delta_b)]
        self.weights = [w - eta * (tdw / len(batch)) for w, tdw in zip(self.weights, total_delta_w)]
        

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

        error = sigmoid_prime(zs[-1]) * self.cost_derivative(activations[-1], y)

        delta_b[-1] = error
        delta_w[-1] = [e * activations[-2] for e in error]
        for l in range(2, self.layer_count):
            z = zs[-l]
            w = np.transpose(self.weights[-l + 1])
            error = np.dot(w, error) * sigmoid_prime(z)

            delta_b[-l] = error
            delta_w[-l] = [e * activations[-l-1] for e in error]

        return (delta_b, delta_w)


    def cost_derivative(self, a, y) -> np.array:
        return 2 * (a - y)

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
        return correct_count

