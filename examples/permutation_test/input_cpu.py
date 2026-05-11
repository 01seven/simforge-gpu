"""Permutation test CPU fixture."""

import numpy as np


def run(n_permutations: int = 1000) -> float:
    group_a = np.array([1.0, 2.0, 3.0, 4.0])
    group_b = np.array([2.0, 3.0, 4.0, 5.0])
    observed = np.mean(group_b) - np.mean(group_a)
    combined = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0])
    count = 0
    for _ in range(n_permutations):
        order = np.random.uniform(size=len(combined))
        permuted = combined[np.argsort(order)]
        diff = np.mean(permuted[4:]) - np.mean(permuted[:4])
        count += diff >= observed
    return float(count / n_permutations)


if __name__ == "__main__":
    print(run())
