import numpy as np


def relu(input):
    return np.maximum(0, input)


def relu_derivative(input):
    return np.where(input > 0, 1, 0)


def calculate_loss(activation, prediction):
    return np.mean((activation - prediction) ** 2)
