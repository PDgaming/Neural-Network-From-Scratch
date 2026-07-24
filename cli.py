import argparse
import json
import sys
import numpy as np
from data import load_dataset, Dataset, DataLoader
from model import build_model
from losses import MSE, CategoricalCrossEntropy, BinaryCrossEntropy, Huber
from metrics import Accuracy, MAE, RMSE, R2
from optimizers import SGD, Adam, Momentum, RMSProp
from train import Trainer


LOSS_MAP = {
    "mse": MSE,
    "huber": Huber,
    "binary_crossentropy": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
}

OPTIMIZER_MAP = {
    "sgd": SGD,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}

METRIC_MAP = {
    "accuracy": Accuracy,
    "mae": MAE,
    "rmse": RMSE,
    "r2": R2,
}


def train(args):
    data, target, input_size, output_size, task, num_classes = load_dataset(
        args.dataset, task=args.task
    )
    print(f"Loaded {args.dataset}: {data.shape[0]} samples, {input_size} features, "
          f"{output_size} outputs ({task})")

    with open(args.architecture) as f:
        architecture = json.load(f)

    model = build_model(architecture, input_size, output_size)
    criterion = LOSS_MAP[args.loss]()
    optimizer = OPTIMIZER_MAP[args.optimizer](args.lr)
    metrics = [METRIC_MAP[m]() for m in args.metrics]

    dataset = Dataset(data, target)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer = Trainer(
        model, criterion, optimizer,
        epochs=args.epochs,
        metrics=metrics,
        eval_every=args.eval_every,
        patience=args.patience
    )

    outputs, losses, metric_logs = trainer.fit(loader, data, target)

    if args.save:
        model.save(args.save)
        print(f"Model saved to {args.save}.npz")


def predict(args):
    with open(args.architecture) as f:
        architecture = json.load(f)

    dummy_input = np.zeros((1, args.input_size))
    model = build_model(architecture, args.input_size, args.output_size)
    model.load(args.load)

    values = [float(x) for x in args.input.split(",")]
    x = np.array(values).reshape(1, -1)

    pred = model.predict(x)
    print(f"Prediction: {np.array2string(pred, precision=4, suppress_small=True)}")


def main():
    parser = argparse.ArgumentParser(description="Neural Network CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--task", choices=["classification", "regression"])
    train_parser.add_argument("--architecture", required=True)
    train_parser.add_argument("--loss", default="categorical_crossentropy", choices=list(LOSS_MAP.keys()))
    train_parser.add_argument("--optimizer", default="adam", choices=list(OPTIMIZER_MAP.keys()))
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--patience", type=int, default=None)
    train_parser.add_argument("--metrics", nargs="+", default=["accuracy"], choices=list(METRIC_MAP.keys()))
    train_parser.add_argument("--save", default=None)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--load", required=True)
    predict_parser.add_argument("--architecture", required=True)
    predict_parser.add_argument("--input-size", type=int, required=True)
    predict_parser.add_argument("--output-size", type=int, required=True)
    predict_parser.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
