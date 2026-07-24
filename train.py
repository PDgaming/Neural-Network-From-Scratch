class Trainer:
    def __init__(self, model, criterion, optimizer, epochs, metrics=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics or []

    def fit(self, loader, eval_data=None, eval_target=None):
        losses = []
        outputs = []
        metric_logs = {metric.__class__.__name__: [] for metric in self.metrics}

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            for x_batch, y_batch in loader:
                prediction = self.model.forward(x_batch)

                loss = self.criterion.forward(prediction, y_batch)
                epoch_loss += loss
                batches += 1

                gradient = self.criterion.backward()
                self.model.backward(gradient)

                self.optimizer.step()

                for layer in self.model.layers:
                    if layer.parameters() is not None:
                        self.optimizer.update(layer)

            epoch_loss /= batches
            losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                if eval_data is not None:
                    full_pred = self.model.forward(eval_data)
                    outputs.append(full_pred.copy())

                    if eval_target is not None:
                        for metric in self.metrics:
                            value = metric.forward(full_pred, eval_target)
                            metric_logs[metric.__class__.__name__].append(value)

                msg = f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}"
                if eval_target is not None and self.metrics:
                    for name, values in metric_logs.items():
                        if values:
                            msg += f", {name}: {values[-1]:.6f}"
                print(msg)

            if eval_data is not None:
                full_pred = self.model.forward(eval_data)
                outputs.append(full_pred.copy())

                if eval_target is not None:
                    for metric in self.metrics:
                        value = metric.forward(full_pred, eval_target)
                        metric_logs[metric.__class__.__name__].append(value)

            if epoch_loss < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break

        return outputs, losses, metric_logs
