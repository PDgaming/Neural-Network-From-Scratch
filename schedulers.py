import math


class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.epoch = 0

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None, metric=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
        self.optimizer.learning_rate = self.get_lr()


class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.epoch // self.step_size))


class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** self.epoch)


class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, T_max=50, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.epoch / self.T_max)
        ) / 2


class ReduceOnPlateau(Scheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=1e-6):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
        self.wait = 0

    def get_lr(self):
        return self.optimizer.learning_rate

    def step(self, epoch=None, metric=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        if metric is None:
            return

        if metric < self.best:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.optimizer.learning_rate * self.factor, self.min_lr)
                self.optimizer.learning_rate = new_lr
                self.wait = 0
