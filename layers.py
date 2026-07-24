import numpy as np


class Dense:
    def __init__(self, input, output):
        self.weight = np.random.randn(input, output) * np.sqrt(2 / input)
        self.bias = np.ones(output) * 0.01

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
