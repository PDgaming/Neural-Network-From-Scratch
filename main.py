from losses import MSE
from activations import Relu
from layers import Dense
from model import Sequential
from data import data, target, test_data
from train import Trainer
from history import History
import numpy as np


def init():
    criterion = MSE()
    model = Sequential(
        [
            Dense(1, 16),
            Relu(),
            Dense(16, 1),
        ]
    )
    history = History()

    return model, criterion, history


def train(model, criterion, history):
    trainer = Trainer(model, criterion, learning_rate=0.01, epochs=100000)
    outputs, losses = trainer.fit(data, target)
    model.save("3x+2 model")

    history.update(
        data,
        target,
        outputs,
        losses,
    )
    history.plot_loss()
    history.plot_prediction()


def load(model):
    model.load("3x+2 model.npz")


def predict(model, input):
    prediction = model.predict(input)
    print(f"Data: {input}, \nPrediction: {prediction}")


model, criterion, history = init()
# train(model, criterion, history)
load(model)
predict(model, test_data)
