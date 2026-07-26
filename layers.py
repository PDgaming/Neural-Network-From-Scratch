import numpy as np
from initializers import He


class Dense:
    def __init__(self, input, output, initializer=None):
        init = initializer or He()
        self.weight = init.weights(input, output)
        self.bias = init.biases(output)

    def forward(self, input):
        self.input = input

        z = (self.input @ self.weight) + self.bias

        return z

    def backward(self, gradient):
        batch_size = self.input.shape[0]

        dL_dw = self.input.T @ gradient
        dL_db = gradient.sum(axis=0)

        self.dw = dL_dw
        self.db = dL_db

        dL_dx = gradient @ self.weight.T

        return dL_dx

    def parameters(self):
        return {
            "weight": self.weight,
            "bias": self.bias,
        }
