from losses import MSE
from activations import Relu
from layers import Dense
from model import Sequential
from data import data, target, test_data, Dataset, DataLoader
from train import Trainer
from history import History


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
    dataset = Dataset(data, target)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    return model, criterion, history, dataset, train_loader, test_loader


def train(model, criterion, history, loader, eval_data):
    trainer = Trainer(model, criterion, learning_rate=0.01, epochs=100)
    outputs, losses = trainer.fit(loader, eval_data)
    model.save("3x+2 model")

    history.update(
        eval_data,
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


model, criterion, history, dataset, train_loader, test_loader = init()

train(model, criterion, history, train_loader, data)
# load(model)
# predict(model, [12])
