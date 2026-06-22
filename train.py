class Trainer:
    def __init__(self, model, criterion, learning_rate, epochs):
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, input, target):
        losses = []
        outputs = []

        for epoch in range(self.epochs):
            output = self.model.forward(input)

            loss = self.criterion.forward(output, target)
            if loss < 1e-6:
                break
            losses.append(loss)
            if (epoch + 1) % 10 == 0:
                outputs.append(output.copy())
                print(f"Epoch {epoch+1}, Loss: {loss}")

            gradient = self.criterion.backward()

            self.model.backward(gradient)

            self.model.step(self.learning_rate)

        return outputs, losses
