import numpy as np


class Relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0.1, self.input)

    def backward(self, gradient):
        mask = self.input > 0
        return gradient * mask


class MSE:
    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction

        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.prediction.size
