from utils import MSE, Relu
import numpy as np


class Dense:
    def __init__(self, input, output):
        self.weight = np.random.randn(input, output) * np.sqrt(2 / input)
        self.bias = np.ones(output) * 0.01

    def forward(self, input):
        self.input = input

        z = (self.input @ self.weight) + self.bias

        return z

    def backward(self, gradient):
        batch_size = self.input.shape[0]

        dL_dw = (self.input.T @ gradient) / batch_size
        dL_db = np.mean(gradient, axis=0)

        self.dw = dL_dw
        self.db = dL_db

        dL_dx = gradient @ self.weight.T

        return dL_dx

    def step(self, learning_rate):
        self.weight -= learning_rate * self.dw
        self.bias -= learning_rate * self.db


x = np.array(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]
)
target = np.array(
    [
        [5.0],
        [8.0],
        [11.0],
        [14.0],
    ]
)
LR = 0.01
epoch = 10000

MSE = MSE()

network = [
    Dense(1, 32),
    Relu(),
    Dense(32, 16),
    Relu(),
    Dense(16, 1),
]


def train(input, target, learning_rate, epochs):
    for epoch in range(epochs):
        output = input
        for layer in network:

            output = layer.forward(output)

        loss = MSE.forward(output, target)
        print(f"Epoch {epoch+1}, Loss: {loss}, Current prediction: {output}")

        gradient = MSE.backward()

        for layer in reversed(network):
            gradient = layer.backward(gradient)

        for layer in network:
            if isinstance(layer, Dense):
                layer.step(learning_rate)


train(x, target, LR, epoch)
