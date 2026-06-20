import numpy as np


def relu(input):
    if input < 0:
        return 0
    else:
        return input


def calculate_loss(input, prediction):
    loss = prediction - input
    square = loss**2
    return loss, square
