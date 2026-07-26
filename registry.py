from layers import Dense
from activations import Relu, Sigmoid, Tanh, LeakyRelu, Softmax
from initializers import He, Xavier, Uniform
from losses import MSE, CategoricalCrossEntropy, BinaryCrossEntropy, Huber
from metrics import Accuracy, MAE, RMSE, R2
from optimizers import SGD, Adam, Momentum, RMSProp
from schedulers import StepLR, ExponentialLR, CosineAnnealingLR, ReduceOnPlateau


LAYER_REGISTRY = {
    "Dense": Dense,
    "Relu": Relu,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "LeakyRelu": LeakyRelu,
    "Softmax": Softmax,
}

INITIALIZER_REGISTRY = {
    "he": He,
    "xavier": Xavier,
    "uniform": Uniform,
}

LOSS_REGISTRY = {
    "mse": MSE,
    "huber": Huber,
    "binary_crossentropy": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
}

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}

METRIC_REGISTRY = {
    "accuracy": Accuracy,
    "mae": MAE,
    "rmse": RMSE,
    "r2": R2,
}

SCHEDULER_REGISTRY = {
    "steplr": StepLR,
    "exponentiallr": ExponentialLR,
    "cosineannealinglr": CosineAnnealingLR,
    "reduceonplateau": ReduceOnPlateau,
}
