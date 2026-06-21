from utils import *
import numpy as np


class Dense:
    def __init__(self, weights, baiases):
        self.weight = weights
        self.bias = baiases

    def forward(self, input):
        self.input = input

        z = np.dot(self.input, self.weight) + self.bias

        return z

    def backward(self, gradient):
        dL_dw = gradient * self.input
        dL_db = gradient

        self.dw = dL_dw
        self.db = dL_db

        dL_dx = gradient * self.weight

        return dL_dx

    def step(self, learning_rate):
        self.weight -= learning_rate * self.dw
        self.bias -= learning_rate * self.db


x = np.array([2.0])
w = np.array([3.0])
b = np.array([1.0])
target = np.array([10.0])
LR = 0.1
epoch = 100


dense = Dense(w, b)
relu = Relu()
MSE = MSE()


def train(input, target, learning_rate, epochs):
    for epoch in range(epochs):
        z = dense.forward(input)
        a = relu.forward(z)

        loss = MSE.forward(a, target)
        print(f"Epoch {epoch+1}, Loss: {loss}, Current prediction: {a}")

        dL_da = MSE.backward()

        dL_dz = relu.backward(dL_da)
        dL_dx = dense.backward(dL_dz)

        dense.step(learning_rate)


train(x, target, LR, epoch)
