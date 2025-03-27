import torch
from torch import Tensor

from util import lasso_loss, run_algorithm, _device


def prox_solver(A: Tensor, b: Tensor, mu: float) -> tuple[float, Tensor, int]:
    m, n = A.shape
    dtype = A.dtype
    mu_list = list(reversed([(5 ** i) * mu for i in range(6)]))
    last_idx = len(mu_list) - 1
    f_last = torch.inf
    x = torch.zeros(n, device=_device, dtype=dtype)
    total_iter = 0
    tol = 1e-8

    grad_last = torch.zeros(n, device=_device, dtype=dtype)
    x_last = torch.zeros(n, device=_device, dtype=dtype)
    zeros = torch.zeros_like(x)

    @torch.compile
    def iterate(x: Tensor, grad_last: Tensor, x_last: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # apply grad step
        grad = A.T @ (A @ x - b)
        s = x - x_last
        y = grad - grad_last
        if total_iter > 1:
            step_size = torch.dot(s, s) / torch.dot(s, y)
        else:
            step_size = 0.01
        x_last = x.clone()
        x -= step_size * grad

        # apply prox
        sign = torch.sign(x)
        x = sign * torch.maximum(torch.abs(x) - mu * step_size, zeros)
        x[torch.abs(x) < 1e-4] = 0

        return x, grad, x_last

    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            total_iter += 1
            inner_iter += 1
            if inner_iter > 50 and k != last_idx:
                break

            x, grad_last, x_last = iterate(x, grad_last, x_last)

            f_new = lasso_loss(A, b, x, mu)

            if abs(f_last - f_new) < tol:
                break

            f_last = f_new

    return f_last, x, total_iter


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, prox_solver, benchmark=True, dtype=torch.float32)
