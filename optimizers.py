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
        if id(layer) not in self.m:
            self.m[id(layer)] = {
                "m_w": np.zeros_like(layer.weight),
                "m_b": np.zeros_like(layer.bias),
            }
            self.v[id(layer)] = {
                "v_w": np.zeros_like(layer.weight),
                "v_b": np.zeros_like(layer.bias),
            }

        self.t += 1
        m = self.m[id(layer)]
        v = self.v[id(layer)]

        m["m_w"] = self.beta1 * m["m_w"] + (1 - self.beta1) * layer.dw
        m["m_b"] = self.beta1 * m["m_b"] + (1 - self.beta1) * layer.db
        v["v_w"] = self.beta2 * v["v_w"] + (1 - self.beta2) * layer.dw ** 2
        v["v_b"] = self.beta2 * v["v_b"] + (1 - self.beta2) * layer.db ** 2

        m_w_hat = m["m_w"] / (1 - self.beta1 ** self.t)
        m_b_hat = m["m_b"] / (1 - self.beta1 ** self.t)
        v_w_hat = v["v_w"] / (1 - self.beta2 ** self.t)
        v_b_hat = v["v_b"] / (1 - self.beta2 ** self.t)

        layer.weight -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
