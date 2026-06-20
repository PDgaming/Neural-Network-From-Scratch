from utils import *
import numpy as np


def forward_pass(input, weight, bias):
    y = (input * weight) + bias
    activated = relu(y)
    return activated


def back_propagation(input, weight, bias, learning_rate, loss):
    delta_loss = 2 * loss
    delta_y = input
    gradient = delta_loss * delta_y
    gradient = np.clip(np.abs(gradient), 0.0, 1.0)

    new_weight = weight - (learning_rate * (-gradient))
    new_bias = bias - (learning_rate * np.sum(gradient))

    return new_weight, new_bias


def train(input, weight, bias, answer, learning_rate, epoch):
    for x in range(epoch):
        forward_result = forward_pass(input, weight, bias)
        squared_loss = calculate_loss(forward_result, answer)
        print(
            f"Epoch {x+1}, Loss: {squared_loss}, Current prediction: {forward_result}"
        )

        weight, bias = back_propagation(
            input, weight, bias, learning_rate, squared_loss
        )

    return weight, bias


x = np.array([2])
w = np.array([3])
b = np.array([1])
answer = 10
LR = 0.1
epoch = 100

weight, bias = train(x, w, b, answer, LR, epoch)
output = forward_pass(x, weight, bias)
print(f"Output: {output}")
