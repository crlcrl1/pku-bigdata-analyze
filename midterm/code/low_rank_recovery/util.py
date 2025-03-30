import time
from typing import Callable

import numpy as np


def gen_data(m: int, n: int, r: int, sample_rate: float, *, seed: int = 2021):
    p = round(m * n * sample_rate)
    np.random.seed(seed)

    omega = np.random.permutation(m * n)[:p]
    xl = np.random.randn(m, r)
    xr = np.random.randn(n, r)
    A = xl @ xr.T
    M = A.flatten()[omega]

    return A, M, omega


def loss(X: np.ndarray, M: np.ndarray, omega: np.ndarray, mu: float) -> float:
    return np.linalg.norm(X.flatten()[omega] - M) ** 2 + mu * np.linalg.norm(X, ord="nuc")


def run_algorithm(m: int, n: int, r: int, sample_rate: float, mu: float, func: Callable, *,
                  benchmark=False, seed: int = 0) -> None:
    A, M, omega = gen_data(m, n, r, sample_rate, seed=seed)
    f, X, iter_num = func(M, omega, mu, m, n)
    print(f"Optimal value: {f}, Iteration number: {iter_num}")
    print(f"Error: {np.linalg.norm(X - A):.4e}")
    print(f"Rank: {np.linalg.matrix_rank(X)}")
    if benchmark:
        for _ in range(5):
            func(M, omega, mu, m, n)
        start = time.time()
        for _ in range(20):
            func(M, omega, mu, m, n)
        end = time.time()
        print(f"Average runtime: {(end - start) / 20:.6f} sec")
