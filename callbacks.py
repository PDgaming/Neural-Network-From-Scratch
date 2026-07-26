import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class LivePlotter(Callback):
    def __init__(self, eval_every=10):
        self.eval_every = eval_every
        self.loss_line = None
        self.metric_lines = {}
        self.loss_data = []
        self.metric_data = {}
        self.fig = None

    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, (self.ax_loss, self.ax_metrics) = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle("Training Progress")
        (self.loss_line,) = self.ax_loss.plot([], [], color="#2196F3", linewidth=1.5)
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_metrics.set_xlabel("Evaluation Step")
        self.ax_metrics.set_ylabel("Metric Value")
        self.ax_metrics.set_title("Metrics")
        self.ax_metrics.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            self.loss_data.append(loss)
            self.loss_line.set_data(range(len(self.loss_data)), self.loss_data)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        metrics = logs.get("metrics", {})
        for name, value in metrics.items():
            if name not in self.metric_data:
                (line,) = self.ax_metrics.plot([], [], label=name, linewidth=1.5)
                self.metric_lines[name] = line
                self.metric_data[name] = []
            if value is not None:
                self.metric_data[name].append(value)
                self.metric_lines[name].set_data(
                    range(len(self.metric_data[name])),
                    self.metric_data[name],
                )

        if metrics:
            self.ax_metrics.relim()
            self.ax_metrics.autoscale_view()
            self.ax_metrics.legend(loc="best")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show(block=False)
