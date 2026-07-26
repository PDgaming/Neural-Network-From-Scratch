import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import build_model


def load_mnist(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", dtype=np.uint8, skip_header=1)
    labels = data[:, 0]
    pixels = data[:, 1:]
    return labels, pixels


def pixels_to_image(pixel_row):
    return pixel_row.reshape(28, 28)


def show_digit(labels, pixels, index):
    img = pixels_to_image(pixels[index])
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Label: {labels[index]}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_grid(labels, pixels, n_rows, n_cols):
    n = n_rows * n_cols
    indices = np.random.choice(len(labels), size=n, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        idx = indices[i]
        img = pixels_to_image(pixels[idx])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(str(labels[idx]), fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_prediction(label, pixel_row, pred, index):
    predicted_class = int(np.argmax(pred))
    confidence = float(pred[predicted_class]) * 100

    img = pixels_to_image(pixel_row)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1, 1.2]})

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Row: {index}", fontsize=12)
    axes[0].axis("off")

    bars = axes[1].barh(range(len(pred)), pred, color="#4a90d9")
    axes[1].set_yticks(range(len(pred)))
    axes[1].set_yticklabels([str(i) for i in range(len(pred))])
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Prediction Distribution", fontsize=12)
    axes[1].set_xlim(0, 1)

    highlight_color = "#2ecc71" if predicted_class == label else "#e74c3c"
    bars[predicted_class].set_color(highlight_color)
    bars[predicted_class].set_label(f"Pred: {predicted_class}")

    axes[1].legend(fontsize=10)

    fig.suptitle(f"True: {label} | Predicted: {predicted_class} ({confidence:.1f}%)",
                 fontsize=14, fontweight="bold",
                 color="#2ecc71" if predicted_class == label else "#e74c3c")
    plt.tight_layout()
    plt.show()


def predict_digit(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.dataset_file)

    labels, pixels = load_mnist(csv_path)

    index = args.predict

    if index < 0 or index >= len(labels):
        print(f"Index out of range. Valid range: 0 - {len(labels) - 1}")
        return

    with open(args.architecture) as f:
        architecture = json.load(f)

    input_size = pixels.shape[1]
    output_size = len(np.unique(labels))
    model = build_model(architecture, input_size, output_size)
    model.load(args.model)

    mean = pixels.mean(axis=0).astype(np.float32)
    std = pixels.std(axis=0).astype(np.float32) + 1e-8
    x = (pixels[index].astype(np.float32) - mean) / std
    x = x.reshape(1, -1)

    pred = model.predict(x)
    predicted_class = int(np.argmax(pred))
    confidence = float(pred[predicted_class]) * 100
    true_label = int(labels[index])

    print(f"Row: {index} | True: {true_label} | Predicted: {predicted_class} | Confidence: {confidence:.1f}%")
    print(f"Probabilities: {np.array2string(pred, precision=4, suppress_small=True)}")

    show_prediction(true_label, pixels[index], pred, index)


def main():
    parser = argparse.ArgumentParser(description="MNIST CSV Visualizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", type=int, metavar="INDEX",
                       help="Show a single digit at the given index")
    group.add_argument("--grid", action="store_true",
                       help="Show a grid of random digits")
    group.add_argument("--predict", type=int, metavar="INDEX",
                       help="Show digit at INDEX with model prediction overlay")
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of grid rows (default: 5)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Number of grid columns (default: 8)")
    parser.add_argument("--dataset", type=str, default="mnist_test.csv",
                        help="CSV filename in datasets/ (default: mnist_test.csv)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to saved .npz model (required with --predict)")
    parser.add_argument("--architecture", type=str, default=None,
                        help="Path to architecture JSON (required with --predict)")
    parser.add_argument("--dataset-file", type=str, default=None,
                        help="CSV filename in datasets/ for prediction (default: same as --dataset)")

    args = parser.parse_args()

    if args.predict is not None:
        if args.model is None or args.architecture is None:
            parser.error("--predict requires --model and --architecture")
        if args.dataset_file is None:
            args.dataset_file = args.dataset
        predict_digit(args)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.dataset)

    labels, pixels = load_mnist(csv_path)
    print(f"Loaded {len(labels)} samples from {args.dataset}")

    if args.single is not None:
        if args.single < 0 or args.single >= len(labels):
            print(f"Index out of range. Valid range: 0 - {len(labels) - 1}")
            return
        show_digit(labels, pixels, args.single)
    else:
        show_grid(labels, pixels, args.rows, args.cols)


if __name__ == "__main__":
    main()
