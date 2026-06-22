from losses import MSE
from activations import Relu
from layers import Dense
from model import Sequential
from data import data, target, test_data
from train import Trainer


def init():
    criterion = MSE()
    model = Sequential(
        [
            Dense(1, 16),
            Relu(),
            Dense(16, 16),
            Relu(),
            Dense(16, 1),
        ]
    )

    return model, criterion


def train(model, criterion):
    trainer = Trainer(model, criterion, learning_rate=0.01, epochs=100000)
    trainer.fit(data, target)
    model.save("model")


def load(model):
    model.load("model.npz")


def predict(model):
    prediction = model.predict(test_data)
    print(f"Data: {test_data}, \nPrediction: {prediction}")


model, criterion = init()
# train(model, criterion)
load(model)
predict(model)
