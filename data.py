import numpy as np
import csv
import os
import re

OUTPUT_KEYWORDS = {
    "label", "target", "class", "output", "y",
    "category", "answer", "prediction", "ground_truth",
    "truth", "response", "dependent", "outcome",
}


def normalize_headers(header, rows):
    cleaned = [h.strip().lower() for h in header]

    already_normalized = all(
        re.fullmatch(r"input_\d+", h) for h in cleaned if not h.startswith("output_")
    ) and any(h.startswith("output_") for h in cleaned)

    if already_normalized:
        return header, rows

    keyword_matches = [i for i, h in enumerate(cleaned) if h in OUTPUT_KEYWORDS]
    explicit_output = None

    if len(keyword_matches) == 1:
        explicit_output = keyword_matches[0]
    elif len(keyword_matches) > 1:
        names = [header[i] for i in keyword_matches]
        raise ValueError(
            f"Multiple output columns detected by keyword: {names}. "
            "Use target_cols to specify which is the output."
        )

    has_header = True
    if explicit_output is None:
        try:
            [float(v) for v in header]
            has_header = False
        except ValueError:
            pass

    if not has_header:
        new_header = [f"input_{i}" for i in range(len(header) - 1)] + ["output_0"]
        return new_header, rows

    if explicit_output is not None:
        input_indices = [i for i in range(len(header)) if i != explicit_output]
        new_header = [f"input_{i}" for i in range(len(input_indices))] + ["output_0"]
        new_rows = [[row[i] for i in input_indices] + [row[explicit_output]] for row in rows]
        return new_header, new_rows

    input_indices = list(range(len(header) - 1))
    output_index = len(header) - 1
    new_header = [f"input_{i}" for i in range(len(input_indices))] + ["output_0"]
    new_rows = [[row[i] for i in input_indices] + [row[output_index]] for row in rows]
    return new_header, new_rows


def load_dataset(name, task=None, normalize=True, target_cols=None):
    if os.path.exists(name):
        csv_path = name
    else:
        csv_path = os.path.join(os.path.dirname(__file__), "datasets", f"{name}.csv")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [list(row) for row in reader]

    if target_cols is not None:
        cleaned = [h.strip().lower() for h in header]
        output_indices = [cleaned.index(t.lower()) for t in target_cols]
        input_indices = [i for i in range(len(header)) if i not in output_indices]
        new_header = [f"input_{i}" for i in range(len(input_indices))] + [f"output_{i}" for i in range(len(output_indices))]
        new_rows = [[row[i] for i in input_indices] + [row[i] for i in output_indices] for row in rows]
        header, rows = new_header, new_rows
    else:
        header, rows = normalize_headers(header, rows)

    input_cols = [i for i, h in enumerate(header) if h.startswith("input_")]
    output_cols = [i for i, h in enumerate(header) if h.startswith("output_")]

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

    inputs = data[:, input_cols].astype(np.float32)
    raw_targets = data[:, output_cols]

    unique_targets = np.unique(raw_targets)
    if task is None:
        if len(unique_targets) <= 20 and np.all(raw_targets == raw_targets.astype(int)):
            task = "classification"
        else:
            task = "regression"

    if task == "classification":
        if raw_targets.shape[1] == 1:
            num_classes = len(unique_targets)
            targets = np.zeros((len(raw_targets), num_classes))
            for i, val in enumerate(raw_targets[:, 0]):
                targets[i, int(val)] = 1
            output_size = num_classes
        else:
            targets = raw_targets.astype(np.float32)
            output_size = targets.shape[1]
            num_classes = output_size
    else:
        targets = raw_targets.astype(np.float32)
        output_size = targets.shape[1] if targets.ndim > 1 else 1
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        num_classes = None

    if normalize:
        mean = inputs.mean(axis=0)
        std = inputs.std(axis=0) + 1e-8
        inputs = (inputs - mean) / std

    input_size = inputs.shape[1]

    return inputs, targets, input_size, output_size, task, num_classes


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
