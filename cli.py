import argparse
import json
import os
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
    if args.max_samples and args.max_samples < data.shape[0]:
        data = data[:args.max_samples]
        target = target[:args.max_samples]
    print(f"Loaded {args.dataset}: {data.shape[0]} samples, {input_size} features, "
          f"{output_size} outputs ({task})", flush=True)

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

    raw, _, _, _, _, _ = load_dataset(args.dataset, task=args.task, normalize=False)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0) + 1e-8

    if args.row is not None:
        if args.row < 0 or args.row >= raw.shape[0]:
            print(f"Row {args.row} out of range. Dataset has {raw.shape[0]} rows (0-{raw.shape[0]-1}).")
            return
        x = raw[args.row:args.row+1]
        x = (x - mean) / std
        true_label = int(raw[args.row, -1]) if raw.shape[1] > args.input_size else None
        print(f"Row: {args.row}" + (f" | True: {true_label}" if true_label is not None else ""))
    else:
        values = [float(v) for v in args.input.split(",")]
        x = np.array(values).reshape(1, -1)
        x = (x - mean) / std

    pred = model.predict(x)
    predicted_class = int(np.argmax(pred))
    confidence = float(pred[predicted_class]) * 100
    print(f"Prediction: {np.array2string(pred, precision=4, suppress_small=True)}")
    print(f"Predicted class: {predicted_class} ({confidence:.1f}%)")

    if args.visualize:
        if args.row is None:
            print("Visualization requires --row to display the digit image.")
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_script = os.path.join(script_dir, "datasets", "mnist_visualize.py")
        cmd = [
            sys.executable, viz_script, "--predict", str(args.row),
            "--model", args.load,
            "--architecture", args.architecture,
            "--dataset-file", args.dataset + ".csv" if not args.dataset.endswith(".csv") else args.dataset,
        ]
        print(f"\nLaunching visualizer...")
        os.execvp(sys.executable, cmd)


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
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument("--max-samples", type=int, default=None)
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
    predict_parser.add_argument("--input", default=None, help="Comma-separated input values (mutually exclusive with --row)")
    predict_parser.add_argument("--row", type=int, default=None, help="Row index from dataset to use as input (mutually exclusive with --input)")
    predict_parser.add_argument("--task", choices=["classification", "regression"])
    predict_parser.add_argument("--visualize", action="store_true", help="Launch visualizer for MNIST digit display")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        if args.input is None and args.row is None:
            predict_parser.error("one of the arguments --input --row is required")
        if args.input is not None and args.row is not None:
            predict_parser.error("arguments --input and --row are mutually exclusive")
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
