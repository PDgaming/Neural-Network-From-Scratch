from utils import *
import numpy as np


def forward_pass(input, weight, bias):
    z = (input * weight) + bias
    a = relu(z)

    return z, a


def back_propagation(
    x,
    z,
    a,
    answer,
    weight,
    bias,
    learning_rate,
):
    dL_da = 2 * (a - answer)
    da_dz = relu_derivative(z)

    dL_dz = dL_da * da_dz

    dL_dw = np.mean(dL_dz * x)
    dL_db = np.mean(dL_dz)

    weight -= learning_rate * dL_dw
    bias -= learning_rate * dL_db

    return weight, bias


def train(input, weight, bias, answer, learning_rate, epochs):
    for epoch in range(epochs):
        z, a = forward_pass(input, weight, bias)
        loss = calculate_loss(a, answer)
        print(f"Epoch {epoch+1}, Loss: {loss}, Current prediction: {a}")

        weight, bias = back_propagation(
            input, z, a, answer, weight, bias, learning_rate
        )

    return weight, bias


x = np.array([2.0, 3.0, 4.0, 5.0])
w = np.array([3.0])
b = np.array([1.0])
answer = np.array([10.0, 15.0, 20.0, 25.0])
LR = 0.01
epoch = 1000

weight, bias = train(x, w, b, answer, LR, epoch)
output = forward_pass(x, weight, bias)
print(f"Output: {output}")
