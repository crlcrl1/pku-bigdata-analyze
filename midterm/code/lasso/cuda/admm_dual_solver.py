import math
from typing import Tuple

import torch
from torch import Tensor

from util import lasso_loss, run_algorithm, _device


def admm_dual_solver(A: Tensor, b: Tensor, mu: float) -> Tuple[float, Tensor, int]:
    m, n = A.shape
    dtype = A.dtype
    iter_count = 0
    tol = 1e-7

    x = torch.ones(n, device=_device, dtype=dtype)
    z = torch.zeros(n, device=_device, dtype=dtype)

    rho = 25

    inv = torch.linalg.inv(rho * A @ A.T + torch.eye(m, device=_device))
    step_size = (1 + math.sqrt(5)) / 2

    @torch.compile
    def iterate(x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        y = inv @ (A @ (rho * z - x) - b)

        z = x / rho + A.T @ y
        abs_z = torch.abs(z)
        mask = abs_z < mu
        abs_z[mask] = mu
        z = z * (mu / abs_z)

        x = x + step_size * rho * (A.T @ y - z)
        return x, y, z

    while True:
        iter_count += 1

        x, y, z = iterate(x, z)

        if torch.linalg.norm(A.T @ y - z) < tol:
            break

    return lasso_loss(A, b, -x, mu), -x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, admm_dual_solver, benchmark=True, dtype=torch.float32)
