from typing import Literal

import numpy as np
from numpy.typing import NDArray
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


def mosek_solver(
        c: NDArray,
        alpha: NDArray,
        beta: NDArray,
        method: Literal["intpnt", "primalSimplex", "dualSimplex"] = "intpnt",
) -> tuple[NDArray, float, int]:
    m = len(alpha)
    n = len(beta)

    c = c.astype(np.float64).flatten()

    with Model("Optimal Transport") as model:
        pi = model.variable("pi", m * n, Domain.greaterThan(0.0))

        model.constraint(Expr.sum(Expr.reshape(pi, m, n), 0), Domain.equalsTo(alpha))
        model.constraint(Expr.sum(Expr.reshape(pi, m, n), 1), Domain.equalsTo(beta))
        model.objective(ObjectiveSense.Minimize, Expr.dot(pi, c))

        model.setSolverParam("optimizer", method)

        model.solve()
        pi_opt = pi.level().reshape(m, n)
        total_cost = model.primalObjValue()

        iter_count = 0
        if method == "intpnt":
            iter_count = model.getSolverIntInfo("intpntIter")
        elif method == "primalSimplex":
            iter_count = model.getSolverIntInfo("simPrimalIter")
        elif method == "dualSimplex":
            iter_count = model.getSolverIntInfo("simDualIter")

        return pi_opt, total_cost, iter_count
