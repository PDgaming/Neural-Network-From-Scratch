import numpy as np


class MSE:
    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.prediction.size


class MAE:
    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction
        return np.mean(np.abs(prediction - target))

    def backward(self):
        return np.sign(self.prediction - self.target) / self.prediction.size


class Huber:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction
        error = prediction - target
        abs_error = np.abs(error)
        quadratic = abs_error <= self.delta
        loss = np.where(
            quadratic,
            0.5 * error**2,
            self.delta * (abs_error - 0.5 * self.delta),
        )
        return np.mean(loss)

    def backward(self):
        error = self.prediction - self.target
        abs_error = np.abs(error)
        quadratic = abs_error <= self.delta
        grad = np.where(quadratic, error, self.delta * np.sign(error))
        return grad / self.prediction.size


class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def forward(self, prediction, target):
        self.target = target
        self.prediction = np.clip(prediction, self.epsilon, 1 - self.epsilon)
        return -np.mean(
            target * np.log(self.prediction)
            + (1 - target) * np.log(1 - self.prediction)
        )

    def backward(self):
        return (
            (self.prediction - self.target)
            / (self.prediction * (1 - self.prediction))
            / self.prediction.size
        )


class CategoricalCrossEntropy:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def forward(self, prediction, target):
        self.target = target
        self.prediction = np.clip(prediction, self.epsilon, 1)
        return -np.mean(np.sum(target * np.log(self.prediction), axis=1))

    def backward(self):
        return (self.prediction - self.target) / self.prediction.shape[0]
