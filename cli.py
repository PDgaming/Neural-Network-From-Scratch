import argparse
import json
import sys
import numpy as np
from data import load_dataset, Dataset, DataLoader
from model import build_model
from registry import LOSS_REGISTRY, OPTIMIZER_REGISTRY, METRIC_REGISTRY, SCHEDULER_REGISTRY
from train import Trainer
from history import History


def train(args):
    data, target, input_size, output_size, task, num_classes = load_dataset(
        args.dataset, task=args.task
    )
    print(f"Loaded {args.dataset}: {data.shape[0]} samples, {input_size} features, "
          f"{output_size} outputs ({task})")

    with open(args.architecture) as f:
        architecture = json.load(f)

    model = build_model(architecture, input_size, output_size)
    criterion = LOSS_REGISTRY[args.loss]()
    optimizer = OPTIMIZER_REGISTRY[args.optimizer](args.lr)
    metrics = [METRIC_REGISTRY[m]() for m in args.metrics]

    scheduler = None
    if args.scheduler:
        params = json.loads(args.scheduler_args) if args.scheduler_args else {}
        scheduler = SCHEDULER_REGISTRY[args.scheduler](optimizer, **params)

    dataset = Dataset(data, target)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer = Trainer(
        model, criterion, optimizer,
        epochs=args.epochs,
        metrics=metrics,
        eval_every=args.eval_every,
        patience=args.patience,
        scheduler=scheduler,
    )

    outputs, losses, metric_logs = trainer.fit(loader, data, target)

    if args.save:
        model.save(args.save)
        print(f"Model saved to {args.save}.npz")

    if args.plot:
        history = History()
        history.update(data, target, outputs, losses, metrics=metric_logs)
        history.plot_loss()
        history.plot_metrics()
        history.plot_prediction()


def predict(args):
    with open(args.architecture) as f:
        architecture = json.load(f)

    model = build_model(architecture, args.input_size, args.output_size)
    model.load(args.load)

    values = [float(x) for x in args.input.split(",")]
    x = np.array(values).reshape(1, -1)

    raw, _, _, _, _, _ = load_dataset(args.dataset, task=args.task, normalize=False)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0) + 1e-8
    x = (x - mean) / std

    pred = model.predict(x)
    print(f"Prediction: {np.array2string(pred, precision=4, suppress_small=True)}")


def main():
    parser = argparse.ArgumentParser(description="Neural Network CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--task", choices=["classification", "regression"])
    train_parser.add_argument("--architecture", required=True)
    train_parser.add_argument("--loss", default="categorical_crossentropy", choices=list(LOSS_REGISTRY.keys()))
    train_parser.add_argument("--optimizer", default="adam", choices=list(OPTIMIZER_REGISTRY.keys()))
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--patience", type=int, default=None)
    train_parser.add_argument("--metrics", nargs="+", default=["accuracy"], choices=list(METRIC_REGISTRY.keys()))
    train_parser.add_argument("--scheduler", default=None, choices=list(SCHEDULER_REGISTRY.keys()))
    train_parser.add_argument("--scheduler-args", default=None, help='JSON string of scheduler kwargs, e.g. \'{"step_size": 100, "gamma": 0.5}\'')
    train_parser.add_argument("--save", default=None)
    train_parser.add_argument("--plot", action="store_true")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--load", required=True)
    predict_parser.add_argument("--architecture", required=True)
    predict_parser.add_argument("--dataset", required=True)
    predict_parser.add_argument("--input-size", type=int, required=True)
    predict_parser.add_argument("--output-size", type=int, required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--task", choices=["classification", "regression"])

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
