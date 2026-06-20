import numpy as np


def relu(input):
    return np.where(input < 0, input, input)


def calculate_loss(result, prediction):
    return np.mean((result - prediction) ** 2)
