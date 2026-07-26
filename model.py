import numpy as np
from layers import Dense
from registry import LAYER_REGISTRY, INITIALIZER_REGISTRY


def build_model(architecture, input_size, output_size):
    layers = []
    prev_units = input_size

    for spec in architecture:
        layer_type = spec["type"]

        if layer_type not in LAYER_REGISTRY:
            raise ValueError(
                f"Unknown layer type: {layer_type}. Available: {list(LAYER_REGISTRY.keys())}"
            )

        if layer_type == "Dense":
            out = spec.get("units", output_size)
            init_name = spec.get("init", "he")
            if init_name not in INITIALIZER_REGISTRY:
                raise ValueError(
                    f"Unknown initializer: {init_name}. Available: {list(INITIALIZER_REGISTRY.keys())}"
                )
            initializer = INITIALIZER_REGISTRY[init_name]()
            layers.append(Dense(prev_units, out, initializer=initializer))
            prev_units = out
        else:
            layers.append(LAYER_REGISTRY[layer_type]())

    return Sequential(layers)


class Sequential:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def predict(self, x):
        x = np.atleast_2d(x)
        output = self.forward(x)
        if output.shape[0] == 1:
            output = output.squeeze()
        return output

    def save(self, path):
        data = {}
        dense_num = 0

        for layer in self.layers:
            params = layer.parameters()

            if params is None:
                continue

            data[f"dense{dense_num}_weight"] = params["weight"]
            data[f"dense{dense_num}_bias"] = params["bias"]

            dense_num += 1

        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)

        dense_num = 0

        for layer in self.layers:
            params = layer.parameters()

            if params is None:
                continue

            layer.weight = data[f"dense{dense_num}_weight"]
            layer.bias = data[f"dense{dense_num}_bias"]

            dense_num += 1
