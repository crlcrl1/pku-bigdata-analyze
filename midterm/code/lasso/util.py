from typing import Tuple, Callable
import time

import numpy as np
from numpy.typing import NDArray
import scipy
import numba


def gen_data(m: int, n: int, *, density: float, seed: int = 0) -> Tuple[NDArray, NDArray, NDArray]:
    np.random.seed(seed)
    A = np.random.randn(m, n)
    u = scipy.sparse.random(n, 1, density=density, data_rvs=np.random.randn).toarray().flatten()
    b = A @ u
    return A, u, b


def lasso_loss(A: NDArray, b: NDArray, x: NDArray, mu: float) -> float:
    return np.linalg.norm(A @ x - b) ** 2 + mu * np.linalg.norm(x, ord=1)


def run_algorithm(m: int, n: int, density: float, seed: int, mu: float, func: Callable, *, benchmark=False) -> None:
    A, u, b = gen_data(m, n, density=density, seed=seed)
    f, x, iter_num = func(A, b, mu)
    print(f"Optimal value: {f}, Iteration number: {iter_num}")
    print(f"Error: {np.linalg.norm(x - u):.4e}")
    if benchmark:
        start = time.time()
        for _ in range(100):
            func(A, b, mu)
        end = time.time()
        print(f"Average runtime: {(end - start) / 100:.6f} sec")
