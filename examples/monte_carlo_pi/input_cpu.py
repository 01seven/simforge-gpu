"""Monte Carlo pi estimate CPU fixture."""

import numpy as np


def run(n_samples: int = 10000) -> float:
    x = np.random.uniform(size=n_samples)
    y = np.random.uniform(size=n_samples)
    inside = x * x + y * y <= 1.0
    return float(4.0 * np.mean(inside))


if __name__ == "__main__":
    print(run())
