import numpy as np


def _to_classes(prediction, target):
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        predicted = np.argmax(prediction, axis=1)
        if target.ndim > 1 and target.shape[1] > 1:
            target = np.argmax(target, axis=1)
    else:
        predicted = (prediction > 0.5).astype(int).squeeze()
    return predicted, target


class Accuracy:
    def forward(self, prediction, target):
        predicted, target = _to_classes(prediction, target)
        return np.mean(predicted == target)


class Precision:
    def forward(self, prediction, target):
        predicted, target = _to_classes(prediction, target)
        classes = np.unique(np.concatenate([predicted, target]))
        per_class = []
        for c in classes:
            tp = np.sum((predicted == c) & (target == c))
            fp = np.sum((predicted == c) & (target != c))
            per_class.append(tp / (tp + fp + 1e-10))
        return np.mean(per_class)


class Recall:
    def forward(self, prediction, target):
        predicted, target = _to_classes(prediction, target)
        classes = np.unique(np.concatenate([predicted, target]))
        per_class = []
        for c in classes:
            tp = np.sum((predicted == c) & (target == c))
            fn = np.sum((predicted != c) & (target == c))
            per_class.append(tp / (tp + fn + 1e-10))
        return np.mean(per_class)


class F1Score:
    def forward(self, prediction, target):
        predicted, target = _to_classes(prediction, target)
        classes = np.unique(np.concatenate([predicted, target]))
        per_class = []
        for c in classes:
            tp = np.sum((predicted == c) & (target == c))
            fp = np.sum((predicted == c) & (target != c))
            fn = np.sum((predicted != c) & (target == c))
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            per_class.append(2 * p * r / (p + r + 1e-10))
        return np.mean(per_class)


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
