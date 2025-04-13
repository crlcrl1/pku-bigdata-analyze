import numpy as np
from numpy.typing import NDArray

from util import run_algorithm


def proxgd_solver(
    M: NDArray, omega: NDArray, mu: float, m: int, n: int
) -> tuple[float, NDArray, int]:
    X = np.ones((m, n))
    step_size = 1 / 2
    tol = 1e-8
    mu_list = list(reversed([mu * 2**i for i in range(10)]))
    last_index = len(mu_list) - 1

    iter_num = 0
    f_last = np.inf
    for k, mu in enumerate(mu_list):
        inner_iter = 0
        while True:
            iter_num += 1
            inner_iter += 1

            if k != last_index and inner_iter > 300:
                break

            # step = step_size
            if k == last_index and step_size > 100:
                step = step_size / np.sqrt(inner_iter - 100)
            else:
                step = step_size

            # apply gradient
            X.flat[omega] -= step * 2 * (X.flat[omega] - M)

            # apply proximal operator
            U, s, Vt = np.linalg.svd(X)
            s = np.maximum(s - mu * step, 0)
            X = U * s @ Vt

            f_new = np.linalg.norm(X.flat[omega] - M) ** 2 + mu * np.sum(s)
            if np.abs(f_new - f_last) < tol:
                f_last = f_new
                break
            f_last = f_new

    return f_last, X, iter_num


if __name__ == "__main__":
    run_algorithm(40, 40, 3, 0.5, 0.001, proxgd_solver, benchmark=True)
