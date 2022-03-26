import numpy
import matplotlib.pyplot as plt

def sigmoid(z):
    denominator = 1 + numpy.exp(-z)
    return 1 / denominator

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    x = [i for i in range(-1000, 1000)]
    y = [(j * j * j) + (j * j) for j in x ]

    y2 = [((3 * j * j) + (2 * j))for j in x]
 
    plt.plot(x, y, label = "x^3 + x^2")
    plt.plot(x, y2, label = "3x^2 + 2x")

    plt.legend()
    plt.show()

