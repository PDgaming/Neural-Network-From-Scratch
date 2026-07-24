import numpy as np
import datasets as ds


def load_dataset(name):
    dataset = getattr(ds, name)
    data = dataset["input"]
    target = dataset["output"]

    # data_mean = data.mean(axis=0)
    # data_std = data.std(axis=0)
    # target_mean = target.mean(axis=0)
    # target_std = target.std(axis=0)

    # data = (data - data_mean) / data_std
    # target = (target - target_mean) / target_std

    return data, target


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
