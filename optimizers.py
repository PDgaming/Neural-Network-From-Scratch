import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError

    def step(self):
        pass


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


class RMSProp(Optimizer):
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer):
        if id(layer) not in self.cache:
            self.cache[id(layer)] = {
                "s_w": np.zeros_like(layer.weight),
                "s_b": np.zeros_like(layer.bias),
            }
        s = self.cache[id(layer)]
        s["s_w"] = self.beta * s["s_w"] + (1 - self.beta) * layer.dw ** 2
        s["s_b"] = self.beta * s["s_b"] + (1 - self.beta) * layer.db ** 2

        layer.weight -= self.learning_rate * layer.dw / (np.sqrt(s["s_w"]) + self.epsilon)
        layer.bias -= self.learning_rate * layer.db / (np.sqrt(s["s_b"]) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        lid = id(layer)
        if lid not in self.m:
            self.m[lid] = {
                "m_w": np.zeros_like(layer.weight),
                "m_b": np.zeros_like(layer.bias),
            }
            self.v[lid] = {
                "v_w": np.zeros_like(layer.weight),
                "v_b": np.zeros_like(layer.bias),
            }

        m = self.m[lid]
        v = self.v[lid]

        one_minus_beta1 = 1 - self.beta1
        one_minus_beta2 = 1 - self.beta2

        m["m_w"] += one_minus_beta1 * (layer.dw - m["m_w"])
        m["m_b"] += one_minus_beta1 * (layer.db - m["m_b"])
        v["v_w"] += one_minus_beta2 * (layer.dw ** 2 - v["v_w"])
        v["v_b"] += one_minus_beta2 * (layer.db ** 2 - v["v_b"])

        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        layer.weight -= self.learning_rate * (m["m_w"] / bc1) / (np.sqrt(v["v_w"] / bc2) + self.epsilon)
        layer.bias -= self.learning_rate * (m["m_b"] / bc1) / (np.sqrt(v["v_b"] / bc2) + self.epsilon)

    def step(self):
        self.t += 1
