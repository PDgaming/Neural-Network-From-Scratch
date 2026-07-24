from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from activations import Relu, Tanh, Sigmoid, Softmax
from layers import Dense
from model import Sequential
from data import load_dataset, Dataset, DataLoader
from train import Trainer
from history import History
from metrics import MAE as MAEMetric, RMSE, R2, Accuracy
from optimizers import SGD, Momentum, Adam
import numpy as np

data, target = load_dataset("iris")

input_size = data.shape[1]
output_size = 3
hidden_size = 16


def init():
    criterion = CategoricalCrossEntropy()
    model = Sequential(
        [
            Dense(input_size, hidden_size),
            Relu(),
            Dense(hidden_size, output_size),
            Softmax(),
        ]
    )
    history = History()
    dataset = Dataset(data, target)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    return model, criterion, history, dataset, train_loader, test_loader


def train(model, criterion, history, loader, eval_data, eval_target):
    optimizer = Adam(learning_rate=0.01)
    metrics = [Accuracy(), MSE()]
    trainer = Trainer(model, criterion, optimizer, epochs=10000, metrics=metrics)
    outputs, losses, metric_logs = trainer.fit(loader, eval_data, eval_target)
    model.save("iris model")

    history.update(
        eval_data,
        eval_target,
        outputs,
        losses,
        metrics=metric_logs,
    )
    history.plot_loss()
    history.plot_metrics()
    history.plot_prediction()


def load(model, name):
    model.load(name)


def predict(model, input):
    prediction = model.predict(input)
    # print(f"Data: {input}, \nPrediction: {prediction}")
    print(
        f"Data: {input}, \nPrediction: {np.array2string(prediction, precision=4, suppress_small=True)}"
    )


model, criterion, history, dataset, train_loader, test_loader = init()

train(model, criterion, history, train_loader, data, target)
# load(model, "iris model.npz")
predict(model, test_loader.dataset[69][0])
