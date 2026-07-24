import numpy as np


class Sequential:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def step(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def predict(self, x):
        return self.forward(x)

    def save(self, path):
        data = {}

        dense_num = 0

        for layer in self.layers:
            params = layer.parameters()

            if params is None:
                continue

            data[f"dense{dense_num}_weight"] = params["weight"]
            data[f"dense{dense_num}_bias"] = params["bias"]

            dense_num += 1

        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)

        dense_num = 0

        for layer in self.layers:
            params = layer.parameters()

            if params is None:
                continue

            layer.weight = data[f"dense{dense_num}_weight"]
            layer.bias = data[f"dense{dense_num}_bias"]

            dense_num += 1
