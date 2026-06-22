import numpy as np


class MSE:
    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction

        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.prediction.size
