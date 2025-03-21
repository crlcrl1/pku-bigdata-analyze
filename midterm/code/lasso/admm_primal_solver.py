from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from util import lasso_loss, run_algorithm


def admm_primal_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    m, n = A.shape
    y = np.zeros(n)
    z = np.zeros(n)
    rho = 3
    iter_count = 0
    tol = 1e-8

    inv = np.linalg.inv(A.T @ A + rho * np.eye(n))
    step_size = (1 + np.sqrt(5)) / 2
    temp = A.T @ b

    while True:
        iter_count += 1

        # update x
        x = inv @ (temp + rho * z - y)

        # update z
        z = x + y / rho
        sign = np.sign(z)
        z = sign * np.maximum(np.abs(z) - mu / rho, 0)

        x[np.abs(x) < 1e-5] = 0
        z[np.abs(z) < 1e-5] = 0

        # update y
        y = y + step_size * rho * (x - z)

        if np.linalg.norm(x - z) < tol:
            break

    return lasso_loss(A, b, x, mu), x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, admm_primal_solver)
