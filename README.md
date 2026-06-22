# Neural Network From Scratch

A feedforward neural network implemented from scratch using only NumPy — no deep learning frameworks. Trains a 2-layer network (1 → 16 → 1) to learn the linear function `y = 3x + 2` via manual backpropagation and SGD.

## Project Structure

| File             | Purpose                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `model.py`       | `Sequential` container that chains layers for forward/backward pass |
| `layers.py`      | `Dense` fully-connected layer with He initialization                |
| `activations.py` | `Relu` activation (leaky forward, hard mask backward)               |
| `losses.py`      | `MSE` loss function                                                 |
| `train.py`       | `Trainer` with training loop and early stopping                     |
| `data.py`        | Synthetic `y = 3x + 2` dataset                                      |
| `history.py`     | `History` tracker with loss/prediction plots                        |
| `main.py`        | Entry point — builds, trains, saves, loads, and predicts            |

## How It Works

```
Input (x) → Dense(1→16) → ReLU → Dense(16→1) → Output (ŷ) → MSE Loss
```

- Forward pass computes predictions through the network
- Backward pass computes gradients via the chain rule
- SGD updates weights and biases each epoch
- Training stops early when loss < 1e-6

## Setup

Requires Python ≥ 3.12 and matplotlib.

```bash
pip install matplotlib
```

Or using uv:

```bash
uv sync
```

## Usage

```bash
python main.py
```

Trains the model (uncomment `main.py:50`), saves weights to `3x+2 model.npz`, and plots loss + prediction curves. By default it loads the pre-trained model and runs inference.
