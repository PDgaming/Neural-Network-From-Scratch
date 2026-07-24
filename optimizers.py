import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    def update(self, layer):
        layer.weight -= self.learning_rate * layer.dw
        layer.bias -= self.learning_rate * layer.db


class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity = {}  # keyed by layer id

    def update(self, layer):
        if id(layer) not in self.velocity:
            self.velocity[id(layer)] = {
                "v_w": np.zeros_like(layer.weight),
                "v_b": np.zeros_like(layer.bias),
            }
        v = self.velocity[id(layer)]
        v["v_w"] = self.beta * v["v_w"] + layer.dw
        v["v_b"] = self.beta * v["v_b"] + layer.db

        layer.weight -= self.learning_rate * v["v_w"]
        layer.bias -= self.learning_rate * v["v_b"]
