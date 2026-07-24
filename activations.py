import numpy as np


class Relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)

    def backward(self, gradient):
        mask = self.input > 0
        return gradient * mask

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None
