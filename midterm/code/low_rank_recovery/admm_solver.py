import numpy as np
from numpy.typing import NDArray

from util import run_algorithm


def admm_solver(M: NDArray, omega: NDArray, mu: float, m: int, n: int) -> tuple[float, NDArray, int]:
    X = np.zeros((m, n))
    Y = np.zeros((m, n))
    Z = np.zeros((m, n))

    other = np.array([i for i in range(m * n) if i not in omega])

    tol = 1e-8
    iter_num = 0
    rho = 4 * mu
    step_size = (1 + np.sqrt(5)) / 2

    while True:
        iter_num += 1

        # update X
        X.flat[omega] = (rho * Z.flat[omega] + 2 * M - Y.flat[omega]) / (rho + 2)
        X.flat[other] = (rho * Z.flat[other] - Y.flat[other]) / rho

        # update Z
        Z = X + Y / rho
        U, s, Vt = np.linalg.svd(Z)
        s = np.maximum(s - mu / rho, 0)
        Z = U * s @ Vt

        # update Y
        Y = Y + step_size * rho * (X - Z)

        # check convergence
        if np.linalg.norm(X - Z) < tol:
            break

    return np.linalg.norm(Z.flat[omega] - M) ** 2 + mu * np.linalg.norm(Z, ord="nuc"), Z, iter_num


if __name__ == "__main__":
    run_algorithm(40, 40, 3, 0.3, 0.01, admm_solver)
