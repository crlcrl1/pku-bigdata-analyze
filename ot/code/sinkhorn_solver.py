import numpy as np
from numpy.typing import NDArray


def sinkhorn_algorithm(c: NDArray, alpha: NDArray, beta: NDArray, epsilon=0.1, max_iter=1000, tol=1e-6):
    m, n = c.shape
    K = np.exp(-c / epsilon)
    print(alpha.min(), alpha.max())
    K1 = np.diag(1 / (alpha + 1e-8)) @ K
    u = np.ones(m)
    iter_num = 0

    for _ in range(max_iter):
        iter_num += 1
        u_prev = u.copy()
        u = 1 / (K1 @ (beta / (K.T @ u)))
        if np.linalg.norm(u - u_prev) < tol:
            break

    v = beta / (K.T @ u)
    pi = np.diag(u) @ K @ np.diag(v)
    total_cost = np.sum(pi * c)

    return pi, total_cost, iter_num
