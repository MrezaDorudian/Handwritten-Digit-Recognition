import numpy as np


# implement a sigmoid function,
# that get array as input
def array_sigmoid(array):
    for i in range(len(array)):
        array[i] = sigmoid(array[i])
    return array


# implement a sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# implement a derivative of sigmoid
def sigmoid_derivative(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))


# implement a derivative of sigmoid with an array as input
def array_sigmoid_derivative(array):
    for i in range(len(array)):
        array[i] = sigmoid_derivative(array[i])
    return array


# implement a relu function
def relu(x):
    if x > 0:
        return x
    else:
        return 0


# implement a sigmoid function,
# that get array as input
def array_relu(array):
    for i in range(len(array)):
        array[i] = relu(array[i])
    return array


# implement a derivative of relu
def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0


# implement a derivative of relu with an array as input
def array_relu_derivative(array):
    for i in range(len(array)):
        array[i] = relu_derivative(array[i])
    return array


# implement a tanh function
def tanh(x):
    return np.tanh(x)


# implement a sigmoid function,
# that get array as input
def array_tanh(array):
    for i in range(len(array)):
        array[i] = tanh(array[i])
    return array


# implement a derivative of tanh
def tanh_derivative(x):
    return 1 / np.square(np.cosh(x))


# implement a derivative of tanh with an array as input
def array_tanh_derivative(array):
    for i in range(len(array)):
        array[i] = tanh_derivative(array[i])
    return array
