import numpy as np
import os
import re
import hashlib

OUTPUT_KEYWORDS = {
    "label", "target", "class", "output", "y",
    "category", "answer", "prediction", "ground_truth",
    "truth", "response", "dependent", "outcome",
}


def normalize_headers(header, data):
    cleaned = [h.strip().lower() for h in header]

    already_normalized = all(
        re.fullmatch(r"input_\d+", h) for h in cleaned if not h.startswith("output_")
    ) and any(h.startswith("output_") for h in cleaned)

    if already_normalized:
        return header, data

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
        return new_header, data

    if explicit_output is not None:
        input_indices = [i for i in range(len(header)) if i != explicit_output]
        new_header = [f"input_{i}" for i in range(len(input_indices))] + ["output_0"]
        new_data = np.column_stack([data[:, input_indices], data[:, [explicit_output]]])
        return new_header, new_data

    input_indices = list(range(len(header) - 1))
    output_index = len(header) - 1
    new_header = [f"input_{i}" for i in range(len(input_indices))] + ["output_0"]
    new_data = np.column_stack([data[:, input_indices], data[:, [output_index]]])
    return new_header, new_data


def _get_cache_path(csv_path, task, normalize, target_cols):
    mtime = os.path.getmtime(csv_path)
    key = f"{csv_path}|{mtime}|{task}|{normalize}|{target_cols}"
    cache_dir = os.path.join(os.path.dirname(csv_path), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = hashlib.md5(key.encode()).hexdigest() + ".npz"
    return os.path.join(cache_dir, cache_name)


def _load_from_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path, allow_pickle=True)
    return (
        cached["inputs"],
        cached["targets"],
        int(cached["meta"][0]),
        int(cached["meta"][1]),
        str(cached["meta"][2]),
        int(cached["meta"][3]) if int(cached["meta"][3]) != -1 else None,
    )


def _save_to_cache(cache_path, inputs, targets, input_size, output_size, task, num_classes):
    meta = np.array([
        input_size,
        output_size,
        task,
        num_classes if num_classes is not None else -1,
    ], dtype=object)
    np.savez(cache_path, inputs=inputs, targets=targets, meta=meta)


def load_dataset(name, task=None, normalize=True, target_cols=None):
    if os.path.exists(name):
        csv_path = name
    else:
        csv_path = os.path.join(os.path.dirname(__file__), "datasets", f"{name}.csv")

    cache_path = _get_cache_path(csv_path, task, normalize, target_cols)
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    with open(csv_path, "r") as f:
        header = [h.strip() for h in f.readline().split(",")]
        data = np.loadtxt(f, delimiter=",", dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if target_cols is not None:
        cleaned = [h.strip().lower() for h in header]
        output_indices = [cleaned.index(t.lower()) for t in target_cols]
        input_indices = [i for i in range(len(header)) if i not in output_indices]
        new_header = [f"input_{i}" for i in range(len(input_indices))] + [f"output_{i}" for i in range(len(output_indices))]
        data = np.column_stack([data[:, input_indices], data[:, output_indices]])
        header = new_header
    else:
        header, data = normalize_headers(header, data)

    input_cols = [i for i, h in enumerate(header) if h.startswith("input_")]
    output_cols = [i for i, h in enumerate(header) if h.startswith("output_")]

    string_cols = {}
    for col in range(data.shape[1]):
        col_values = data[:, col]
        n = len(col_values)
        sample_indices = np.unique(np.concatenate([
            np.arange(min(3, n)),
            np.arange(max(0, n - 3), n),
        ]))
        try:
            col_values[sample_indices].astype(float)
        except ValueError:
            unique = np.unique(col_values)
            string_cols[col] = {v: str(i) for i, v in enumerate(unique)}

    for col, mapping in string_cols.items():
        data[:, col] = np.array([mapping[v] for v in data[:, col]])

    data = data.astype(np.float32)

    inputs = data[:, input_cols]
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
            targets = np.zeros((len(raw_targets), num_classes), dtype=np.float32)
            targets[np.arange(len(raw_targets)), raw_targets[:, 0].astype(int)] = 1
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

    _save_to_cache(cache_path, inputs, targets, input_size, output_size, task, num_classes)

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

            yield self.dataset.data[chunk], self.dataset.target[chunk]
