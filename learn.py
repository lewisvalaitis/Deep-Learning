from time import time

from network2 import Network
from cost_functions import CrossEntropyCost
from data import get_test_data, get_training_data, get_shifted_data


data = get_training_data()

test_data = get_test_data()
validation_data = data[-10_000:]
training_data = data[:10_000]

print(len(training_data))
net = Network([784, 30, 10], CrossEntropyCost())
start = time()
net.SGD(training_data, 30, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)


print(f"Time to train: { time() - start }")

# Base line speed of training for a nueral net [784, 30, 10], epochs = 30, batch = 20 and eta = 3.0.
# 92.466 | 92.729 | 94.744 