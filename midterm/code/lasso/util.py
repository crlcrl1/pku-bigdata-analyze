import time
from typing import Callable

import numpy as np
import scipy
from numpy.typing import NDArray


def gen_data(
        m: int, n: int, *, density: float, seed: int = 0
) -> tuple[NDArray, NDArray, NDArray]:
    np.random.seed(seed)
    A = np.random.randn(m, n)
    u = (
        scipy.sparse.random(n, 1, density=density, data_rvs=np.random.randn)
        .toarray()
        .flatten()
    )
    b = A @ u
    return A, u, b


def lasso_loss(A: NDArray, b: NDArray, x: NDArray, mu: float) -> float:
    return np.linalg.norm(A @ x - b) ** 2 + mu * np.linalg.norm(x, ord=1)


def run_algorithm(
        m: int,
        n: int,
        density: float,
        seed: int,
        mu: float,
        func: Callable,
        *,
        benchmark=False,
) -> None:
    A, u, b = gen_data(m, n, density=density, seed=seed)
    x0 = np.zeros(n)
    result = func(x0, A, b, mu)
    x, info = result
    f = info["value"]
    iter_num = info["iterations"]
    print(f"Optimal value: {f:.6f}, Iteration number: {iter_num}")
    max_elem = np.max(np.abs(x))
    print(f"Error: {np.linalg.norm(x - u):.4e}")
    print(f"Sparsity: {np.sum(np.abs(x) > 1e-6 * max_elem) / n:.4f}")
    if benchmark:
        for _ in range(10):
            func(x0, A, b, mu)
        start = time.time()
        for _ in range(40):
            func(x0, A, b, mu)
        end = time.time()
        print(f"Average runtime: {(end - start) / 40:.6f} sec")
