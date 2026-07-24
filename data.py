import numpy as np
import csv
import os


def load_dataset(name):
    csv_path = os.path.join(os.path.dirname(__file__), "datasets", f"{name}.csv")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    mappings = {}
    for col in range(len(header)):
        values = [rows[r][col] for r in range(len(rows))]
        try:
            [float(v) for v in values]
        except ValueError:
            unique = sorted(set(values))
            mappings[col] = {v: i for i, v in enumerate(unique)}

    for row in rows:
        for col in mappings:
            row[col] = mappings[col][row[col]]

    data = np.array([list(map(float, row)) for row in rows])
    input_cols = [i for i, h in enumerate(header) if h.startswith("input_")]
    output_cols = [i for i, h in enumerate(header) if h.startswith("output_")]

    targets = data[:, output_cols].astype(int)
    num_classes = max(len(v) for v in mappings.values())
    one_hot = np.zeros((len(targets), num_classes))
    for i, col in enumerate(mappings):
        one_hot[np.arange(len(targets)), targets[:, i]] = 1

    return data[:, input_cols], one_hot


class Dataset:
    def __init__(self, data, target):
        assert len(data) == len(target)
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        start = 0
        while start < len(indices):
            chunk = indices[start : start + self.batch_size]
            start += self.batch_size

            x_batch = []
            y_batch = []

            for idx in chunk:
                x, y = self.dataset[idx]

                x_batch.append(x)
                y_batch.append(y)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, y_batch
