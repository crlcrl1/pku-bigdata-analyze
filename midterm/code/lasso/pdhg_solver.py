from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from util import lasso_loss, run_algorithm


def pdhg_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    m, n = A.shape
    max_eigval = np.sqrt(np.max(np.linalg.eigvalsh(A @ A.T)))
    step_size_z = 2 / max_eigval
    step_size_x = step_size_z / 4
    mu_list = list(reversed([(5 ** i) * mu for i in range(5)]))
    last_idx = len(mu_list) - 1

    x = np.zeros(n)
    z = np.zeros(m)
    f_last = np.inf
    tol = 1e-8
    iter_count = 0

    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            iter_count += 1
            inner_iter += 1

            if inner_iter > 30 and k != last_idx:
                break

            z = 1 / (1 + step_size_z) * (z + step_size_z * (A @ x - b))
            x = x - step_size_x * A.T @ z
            x = np.sign(x) * np.maximum(np.abs(x) - mu * step_size_x, 0)
            x[np.abs(x) < 1e-5] = 0

            f_new = lasso_loss(A, b, x, mu)
            if abs(f_last - f_new) < tol:
                break
            f_last = f_new

    return f_last, x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, pdhg_solver, benchmark=True)
