from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from util import lasso_loss, run_algorithm


def prox_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    m, n = A.shape
    step_size = 0.01
    mu_list = list(reversed([(5 ** i) * mu for i in range(6)]))
    last_idx = len(mu_list) - 1
    f_last = np.inf
    x = np.zeros(n)
    total_iter = 0

    ata = A.T @ A
    atb = A.T @ b

    grad_last = np.zeros(n)
    x_last = np.zeros(n)

    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            total_iter += 1
            inner_iter += 1
            if inner_iter > 50 and k != last_idx:
                break

            # apply grad step
            grad = ata @ x - atb
            s = x - x_last
            y = grad - grad_last
            if total_iter > 1:
                step_size = np.dot(s, s) / np.dot(s, y)
            x_last = x.copy()
            grad_last = grad
            x -= step_size * grad

            # apply prox
            sign = np.sign(x)
            x = sign * np.maximum(np.abs(x) - mu * step_size, 0)
            x[np.abs(x) < 1e-4] = 0

            f_new = lasso_loss(A, b, x, mu)

            if abs(f_last - f_new) < 1e-8:
                break

            f_last = f_new

    return f_last, x, total_iter


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, prox_solver)
