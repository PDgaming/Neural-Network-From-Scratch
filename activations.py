import numpy as np


class Relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)

    def backward(self, gradient):
        return gradient * (self.input > 0)

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None


class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-np.clip(input, -500, 500)))
        return self.output

    def backward(self, gradient):
        return gradient * self.output * (1 - self.output)

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None


class Tanh:
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, gradient):
        return gradient * (1 - self.output ** 2)

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None


class LeakyRelu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, gradient):
        mask = np.where(self.input > 0, 1, self.alpha)
        return gradient * mask

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None


class Softmax:
    def forward(self, input):
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, gradient):
        s = self.output
        dot = np.sum(gradient * s, axis=1, keepdims=True)
        return s * (gradient - dot)

    def step(self, learning_rate):
        pass

    def parameters(self):
        return None