import numpy as np


class He:
    def weights(self, fan_in, fan_out):
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

    def biases(self, fan_out):
        return np.ones(fan_out) * 0.01


class Xavier:
    def weights(self, fan_in, fan_out):
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_in + fan_out))

    def biases(self, fan_out):
        return np.zeros(fan_out)


class Uniform:
    def __init__(self, limit=None):
        self.limit = limit

    def weights(self, fan_in, fan_out):
        limit = self.limit or np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    def biases(self, fan_out):
        return np.zeros(fan_out)
