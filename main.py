from losses import MSE, BinaryCrossEntropy
from activations import Relu, Tanh, Sigmoid
from layers import Dense
from model import Sequential
from data import load_dataset, Dataset, DataLoader
from train import Trainer
from history import History
from metrics import MAE as MAEMetric, RMSE, R2, Accuracy


data, target = load_dataset("xor_data")

input_size = data.shape[1]
output_size = target.shape[1]
hidden_size = 4


def init():
    criterion = BinaryCrossEntropy()
    model = Sequential(
        [
            Dense(input_size, hidden_size),
            Tanh(),
            Dense(hidden_size, output_size),
            Sigmoid(),
        ]
    )
    history = History()
    dataset = Dataset(data, target)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    return model, criterion, history, dataset, train_loader, test_loader


def train(model, criterion, history, loader, eval_data, eval_target):
    metrics = [Accuracy()]
    trainer = Trainer(model, criterion, learning_rate=0.5, epochs=3000, metrics=metrics)
    outputs, losses, metric_logs = trainer.fit(loader, eval_data, eval_target)
    model.save("xor model")

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
    print(f"Data: {input}, \nPrediction: {prediction}")


model, criterion, history, dataset, train_loader, test_loader = init()

train(model, criterion, history, train_loader, data, target)
# load(model, "xor model.npz")
predict(model, [0, 0])
