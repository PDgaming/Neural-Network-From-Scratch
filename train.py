class Trainer:
    def __init__(self, model, criterion, optimizer, epochs, metrics=None,
                 eval_every=10, patience=None, scheduler=None, callbacks=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics or []
        self.eval_every = eval_every
        self.patience = patience
        self.scheduler = scheduler
        self.callbacks = callbacks or []

    def fit(self, loader, eval_data=None, eval_target=None):
        losses = []
        outputs = []
        metric_logs = {metric.__class__.__name__: [] for metric in self.metrics}

        best_loss = float("inf")
        patience_counter = 0

        for cb in self.callbacks:
            cb.on_train_begin()

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

            current_metrics = {}

            if eval_data is not None and (epoch + 1) % self.eval_every == 0:
                full_pred = self.model.forward(eval_data)
                outputs.append(full_pred.copy())

                if eval_target is not None:
                    for metric in self.metrics:
                        value = metric.forward(full_pred, eval_target)
                        metric_logs[metric.__class__.__name__].append(value)
                        current_metrics[metric.__class__.__name__] = value

                msg = f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}"
                if eval_target is not None and self.metrics:
                    for name, values in metric_logs.items():
                        if values:
                            msg += f", {name}: {values[-1]:.6f}"
                print(msg, flush=True)

                logs = {"epoch": epoch, "loss": epoch_loss, "metrics": current_metrics}
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, logs)

            if self.patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}", flush=True)
                        break

            if self.scheduler is not None:
                self.scheduler.step(epoch, metric=epoch_loss)

        for cb in self.callbacks:
            cb.on_train_end()

        return outputs, losses, metric_logs
