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
        if not hasattr(self, '_dw_buf') or self._dw_buf.shape != (self.input.shape[1], gradient.shape[1]):
            self._dw_buf = np.empty((self.input.shape[1], gradient.shape[1]), dtype=self.weight.dtype)
            self._db_buf = np.empty(gradient.shape[1], dtype=self.bias.dtype)

        np.matmul(self.input.T, gradient, out=self._dw_buf)
        np.sum(gradient, axis=0, out=self._db_buf)
        self.dw = self._dw_buf
        self.db = self._db_buf

        return gradient @ self.weight.T

    def parameters(self):
        return {
            "weight": self.weight,
            "bias": self.bias,
        }
