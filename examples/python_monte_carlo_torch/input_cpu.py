import numpy as np


def estimate_pi(samples: int = 50_000) -> float:
    x = np.random.uniform(0.0, 1.0, samples)
    y = np.random.uniform(0.0, 1.0, samples)
    inside = x * x + y * y <= 1.0
    return float(4.0 * np.mean(inside))


if __name__ == "__main__":
    print(estimate_pi())

