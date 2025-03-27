from typing import Tuple, Callable
import time

import torch
import numpy as np
from torch import Tensor
import scipy

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_data(m: int, n: int, *, density: float, seed: int = 0, dtype=torch.float64) -> Tuple[Tensor, Tensor, Tensor]:
    np.random.seed(seed)
    A = np.random.randn(m, n)
    u = scipy.sparse.random(n, 1, density=density, data_rvs=np.random.randn).toarray().flatten()
    b = A @ u
    A = torch.tensor(A, device=_device, dtype=dtype)
    u = torch.tensor(u, device=_device, dtype=dtype)
    b = torch.tensor(b, device=_device, dtype=dtype)
    return A, u, b


@torch.compile
def lasso_loss(A: Tensor, b: Tensor, x: Tensor, mu: float) -> float:
    return torch.linalg.norm(A @ x - b) ** 2 + mu * torch.linalg.norm(x, ord=1)


def run_algorithm(m: int, n: int, density: float, seed: int, mu: float, func: Callable, *, benchmark=False,
                  dtype=torch.float64) -> None:
    A, u, b = gen_data(m, n, density=density, seed=seed, dtype=dtype)
    f, x, iter_num = func(A, b, mu)
    print(f"Optimal value: {f}, Iteration number: {iter_num}")
    print(f"Error: {torch.linalg.norm(x - u):.4e}")
    max_elem = torch.max(torch.abs(x))
    print(f"Sparsity: {torch.sum(torch.abs(x) > 1e-6 * max_elem) / n:.4f}")
    if benchmark:
        for _ in range(10):
            func(A, b, mu)
        start = time.time()
        for _ in range(90):
            func(A, b, mu)
        end = time.time()
        print(f"Average runtime: {(end - start) / 90:.6f} sec")
