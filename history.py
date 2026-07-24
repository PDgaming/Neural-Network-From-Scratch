import numpy as np
import matplotlib.pyplot as plt


class History:
    def __init__(self):
        self.losses = []
        self.metrics = {}
        self.predictions = []
        self.inputs = None
        self.targets = None

    def update(self, inputs, targets, predictions, losses, metrics=None):
        self.inputs = inputs.copy()
        self.targets = targets.copy()
        self.predictions = predictions
        self.losses = losses
        if metrics:
            self.metrics = metrics

    def plot_loss(self):
        plt.plot(self.losses)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()

    def plot_metrics(self):
        for name, values in self.metrics.items():
            plt.plot(values, label=name)
        plt.xlabel("Evaluation Step (every 10 epochs)")
        plt.ylabel("Metric Value")
        plt.title("Evaluation Metrics")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_prediction(self):
        pred = self.predictions[-1]
        target = self.targets
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze()
        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze()

        if self.inputs.ndim == 1 or self.inputs.shape[1] == 1:
            x = self.inputs.squeeze()
            idx = np.argsort(x)
            plt.scatter(x, target, label="Target")
            plt.plot(x[idx], pred[idx], label="Prediction")
            plt.xlabel("Input")
        else:
            x = np.arange(len(self.inputs))
            plt.scatter(x, target, label="Target")
            plt.plot(x, pred, label="Prediction")
            plt.xlabel("Sample Index")
        plt.ylabel("Output")
        plt.legend()
        plt.show()