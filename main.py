from data import load_dataset, Dataset, DataLoader
from model import build_model
from losses import MSE, CategoricalCrossEntropy, BinaryCrossEntropy, Huber
from metrics import Accuracy, MAE, RMSE, R2
from optimizers import SGD, Adam, Momentum, RMSProp
from train import Trainer
from history import History
import os

# =============================================================================
# CONFIGURE YOUR EXPERIMENT HERE
# =============================================================================

dataset_name = "iris"
task = None

architecture = [
    {"type": "Dense", "units": 16},
    {"type": "Relu"},
    {"type": "Dense", "units": 3},
    {"type": "Softmax"},
]

loss_name = "categorical_crossentropy"

optimizer_name = "adam"
learning_rate = 0.01

epochs = 1000
batch_size = 16
eval_every = 50
patience = 100

metrics_list = ["accuracy"]

save_path = None
load_path = None

# =============================================================================
# RUN
# =============================================================================

data, target, input_size, output_size, task_type, num_classes = load_dataset(
    dataset_name, task=task
)

print(f"Loaded {dataset_name}: {data.shape[0]} samples, {input_size} features, "
      f"{output_size} outputs ({task_type})")

model = build_model(architecture, input_size, output_size)

LOSS_MAP = {
    "mse": MSE,
    "huber": Huber,
    "binary_crossentropy": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
}
criterion = LOSS_MAP[loss_name.lower()]()

OPTIMIZER_MAP = {
    "sgd": lambda: SGD(learning_rate),
    "momentum": lambda: Momentum(learning_rate),
    "rmsprop": lambda: RMSProp(learning_rate),
    "adam": lambda: Adam(learning_rate),
}
optimizer = OPTIMIZER_MAP[optimizer_name.lower()]()

METRIC_MAP = {
    "accuracy": Accuracy,
    "mae": MAE,
    "rmse": RMSE,
    "r2": R2,
}
metrics = [METRIC_MAP[m.lower()]() for m in metrics_list]

dataset = Dataset(data, target)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

history = History()
trainer = Trainer(
    model, criterion, optimizer,
    epochs=epochs,
    metrics=metrics,
    eval_every=eval_every,
    patience=patience
)

outputs, losses, metric_logs = trainer.fit(train_loader, data, target)

if save_path:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.npz")

if load_path:
    model.load(load_path)
    print(f"Model loaded from {load_path}")

history.update(data, target, outputs, losses, metrics=metric_logs)
history.plot_loss()
history.plot_metrics()
history.plot_prediction()
