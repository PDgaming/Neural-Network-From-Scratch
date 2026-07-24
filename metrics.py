import numpy as np


class Accuracy:
    def forward(self, prediction, target):
        if prediction.ndim > 1 and prediction.shape[1] > 1:
            predicted = np.argmax(prediction, axis=1)
        else:
            predicted = (prediction > 0.5).astype(int).squeeze()
        return np.mean(predicted == target)


class MAE:
    def forward(self, prediction, target):
        return np.mean(np.abs(prediction - target))


class RMSE:
    def forward(self, prediction, target):
        return np.sqrt(np.mean((prediction - target) ** 2))


class R2:
    def forward(self, prediction, target):
        ss_res = np.sum((target - prediction) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))