from data import load_dataset, Dataset, DataLoader
from model import build_model
from registry import LOSS_REGISTRY, OPTIMIZER_REGISTRY, METRIC_REGISTRY, SCHEDULER_REGISTRY
from train import Trainer
from history import History
from callbacks import LivePlotter
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

# Set to None to disable scheduling, or use a dict like:
# {"name": "steplr", "step_size": 100, "gamma": 0.5}
# {"name": "exponentiallr", "gamma": 0.95}
# {"name": "cosineannealinglr", "T_max": 100, "eta_min": 1e-5}
# {"name": "reduceonplateau", "factor": 0.1, "patience": 10}
scheduler_config = None

epochs = 1000
batch_size = 16
eval_every = 50
patience = 100

metrics_list = ["accuracy"]

live_plot = True

save_path = None
load_path = None

# =============================================================================
# RUN
# =============================================================================

data, target, input_size, output_size, task_type, num_classes = load_dataset(
    dataset_name, task=task
)

print(
    f"Loaded {dataset_name}: {data.shape[0]} samples, {input_size} features, "
    f"{output_size} outputs ({task_type})", flush=True
)

model = build_model(architecture, input_size, output_size)

criterion = LOSS_REGISTRY[loss_name.lower()]()
optimizer = OPTIMIZER_REGISTRY[optimizer_name.lower()](learning_rate)
metrics = [METRIC_REGISTRY[m.lower()]() for m in metrics_list]

scheduler = None
if scheduler_config is not None:
    name = scheduler_config["name"].lower()
    params = {k: v for k, v in scheduler_config.items() if k != "name"}
    scheduler = SCHEDULER_REGISTRY[name](optimizer, **params)

dataset = Dataset(data, target)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

history = History()

callbacks = []
if live_plot:
    callbacks.append(LivePlotter(eval_every=eval_every))

trainer = Trainer(
    model,
    criterion,
    optimizer,
    epochs=epochs,
    metrics=metrics,
    eval_every=eval_every,
    patience=patience,
    scheduler=scheduler,
    callbacks=callbacks,
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
