from typing import Tuple

import torch
from torch import Tensor

from util import lasso_loss, run_algorithm, _device


def pdhg_solver(A: Tensor, b: Tensor, mu: float) -> Tuple[float, Tensor, int]:
    m, n = A.shape
    dtype = A.dtype
    max_eigval = torch.max(torch.linalg.eigvalsh(A @ A.T)).sqrt()
    step_size_z = 2 / max_eigval
    step_size_x = 1 / (2 * max_eigval)
    mu_list = list(reversed([(5 ** i) * mu for i in range(5)]))
    last_idx = len(mu_list) - 1

    x = torch.zeros(n, device=_device, dtype=dtype)
    z = torch.zeros(m, device=_device, dtype=dtype)
    zeros = torch.zeros_like(x)
    f_last = torch.inf
    tol = 1e-8
    iter_count = 0

    @torch.compile
    def iterate(x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        z = 1 / (1 + step_size_z) * (z + step_size_z * (A @ x - b))
        x -= step_size_x * A.T @ z
        x = torch.sign(x) * torch.maximum(torch.abs(x) - mu * step_size_x, zeros)
        x[torch.abs(x) < 1e-5] = 0
        return x, z

    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            iter_count += 1
            inner_iter += 1

            if inner_iter > 30 and k != last_idx:
                break

            x, z = iterate(x, z)

            f_new = lasso_loss(A, b, x, mu)
            if abs(f_last - f_new) < tol:
                break
            f_last = f_new

    return f_last, x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 1, 0.01, pdhg_solver, benchmark=True, dtype=torch.float32)
