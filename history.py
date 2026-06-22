import matplotlib.pyplot as plt


class History:
    def __init__(self):
        self.losses = []
        self.predictions = []
        self.inputs = None
        self.targets = None

    def update(self, inputs, targets, predictions, losses):
        self.inputs = inputs.copy()
        self.targets = targets.copy()
        self.predictions = predictions
        self.losses = losses

    def plot_loss(self):
        plt.plot(self.losses)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.show()

    def plot_prediction(self):
        plt.scatter(self.inputs, self.targets, label="Target")
        plt.plot(self.inputs, self.predictions[-1], label="Prediction")

        plt.legend()
        plt.show()
