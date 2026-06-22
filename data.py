import numpy as np

data = np.arange(-10, 11, dtype=float).reshape(-1, 1)
target = 3 * data + 2

test_data = np.arange(-12.5, 15.5, 0.5).reshape(-1, 1)
