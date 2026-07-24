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
        plt.scatter(self.inputs, self.targets, label="Target")
        plt.plot(self.inputs, self.predictions[-1], label="Prediction")
        plt.legend()
        plt.show()