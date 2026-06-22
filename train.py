class Trainer:
    def __init__(self, model, criterion, learning_rate, epochs):
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, input, target):
        for epoch in range(self.epochs):
            output = self.model.forward(input)

            loss = self.criterion.forward(output, target)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss}, Current prediction: {output}")

            gradient = self.criterion.backward()

            self.model.backward(gradient)

            self.model.step(self.learning_rate)
