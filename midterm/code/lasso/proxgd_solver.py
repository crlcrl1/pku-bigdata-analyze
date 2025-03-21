from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from util import lasso_loss, run_algorithm


def prox_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    m, n = A.shape
    step_size = 1 / (np.linalg.norm(A, ord=2) ** 2)
    mu_list = list(reversed([(5 ** i) * mu for i in range(6)]))
    last_idx = len(mu_list) - 1
    f_last = np.inf
    x = np.zeros(n)
    total_iter = 0

    ata = A.T @ A
    atb = A.T @ b

    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            total_iter += 1
            inner_iter += 1
            if inner_iter > 150 and k != last_idx:
                break

            # apply grad step
            grad = ata @ x - atb
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
