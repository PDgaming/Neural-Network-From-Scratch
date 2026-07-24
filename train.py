class Trainer:
    def __init__(self, model, criterion, learning_rate, epochs):
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, loader, eval_data=None):
        losses = []
        outputs = []

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
                self.model.step(self.learning_rate)

            epoch_loss /= batches
            losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                if eval_data is not None:
                    full_pred = self.model.forward(eval_data)
                    outputs.append(full_pred.copy())
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")

            if epoch_loss < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break

        return outputs, losses
