"""Bootstrap mean CPU fixture."""

import numpy as np


def run(n_resamples: int = 1000) -> float:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimates = []
    for _ in range(n_resamples):
        indices = np.random.uniform(size=len(data))
        selected = data[(indices * len(data)).astype(int)]
        estimates.append(np.mean(selected))
    return float(np.mean(estimates))


if __name__ == "__main__":
    print(run())
