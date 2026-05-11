"""Normal mean probability CPU fixture."""

import numpy as np


def run(n_samples: int = 10000) -> float:
    draws = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    return float(np.mean(draws > 1.96))


if __name__ == "__main__":
    print(run())
