"""Random walk CPU fixture."""

import numpy as np


def run(n_steps: int = 1000) -> float:
    steps = np.random.binomial(1, 0.5, size=n_steps)
    signed_steps = np.maximum(steps * 2 - 1, -1)
    return float(np.sum(signed_steps))


if __name__ == "__main__":
    print(run())
