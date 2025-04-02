import math

import torch
from torch import Tensor

from util import lasso_loss, run_algorithm, _device


def admm_primal_solver(A: Tensor, b: Tensor, mu: float) -> tuple[float, Tensor, int]:
    m, n = A.shape
    dtype = A.dtype
    y = torch.zeros(n, device=_device, dtype=dtype)
    z = torch.zeros(n, device=_device, dtype=dtype)
    rho = 3
    iter_count = 0
    tol = 1e-8

    inv = torch.linalg.inv(A.T @ A + rho * torch.eye(n, device=_device, dtype=dtype))
    step_size = (1 + math.sqrt(5)) / 2
    temp = A.T @ b

    zeros = torch.zeros_like(z)

    # @torch.compile
    def iterate(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = inv @ (temp + rho * z - y)

        z = x + y / rho
        sign = torch.sign(z)
        z = sign * torch.maximum(torch.abs(z) - mu / rho, zeros)

        x[torch.abs(x) < 1e-5] = 0
        z[torch.abs(z) < 1e-5] = 0

        y = y + step_size * rho * (x - z)
        return x, y, z

    while True:
        iter_count += 1

        x, y, z = iterate(z, y)

        if torch.linalg.norm(x - z) < tol:
            break

    return lasso_loss(A, b, x, mu), x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, admm_primal_solver)
