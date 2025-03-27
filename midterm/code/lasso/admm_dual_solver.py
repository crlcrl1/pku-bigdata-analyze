import numpy as np
from numpy.typing import NDArray

from util import lasso_loss, run_algorithm


def admm_dual_solver(A: NDArray, b: NDArray, mu: float) -> tuple[float, NDArray, int]:
    m, n = A.shape
    iter_count = 0
    tol = 1e-8

    x = np.ones(n)
    z = np.zeros(n)

    rho = 25

    inv = np.linalg.inv(rho * A @ A.T + np.eye(m))
    step_size = (1 + np.sqrt(5)) / 2

    while True:
        iter_count += 1

        # update y
        y = inv @ (A @ (rho * z - x) - b)

        # update z
        z = x / rho + A.T @ y
        abs_z = np.abs(z)
        mask = abs_z < mu
        abs_z[mask] = mu
        z = z * (mu / abs_z)

        # update x
        x = x + step_size * rho * (A.T @ y - z)

        if np.linalg.norm(A.T @ y - z) < tol:
            break

    return lasso_loss(A, b, -x, mu), -x, iter_count


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, admm_dual_solver, benchmark=True)
