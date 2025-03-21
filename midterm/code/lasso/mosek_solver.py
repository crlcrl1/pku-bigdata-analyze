from typing import Tuple

from util import run_algorithm

from mosek.fusion import Model, Domain, Expr, ObjectiveSense
from numpy.typing import NDArray


def mosek_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    m, n = A.shape
    with Model("LASSO") as M:
        x = M.variable("x", n, Domain.unbounded())

        t = M.variable("t", n, Domain.greaterThan(0.0))

        M.constraint(Expr.sub(t, x), Domain.greaterThan(0.0))
        M.constraint(Expr.add(t, x), Domain.greaterThan(0.0))

        r = M.variable("r", m, Domain.unbounded())
        M.constraint(Expr.sub(Expr.mul(A, x), r), Domain.equalsTo(b))

        q = M.variable("q", 1, Domain.greaterThan(0.0))

        M.constraint(Expr.vstack(q, r), Domain.inQCone())

        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.mul(mu, Expr.sum(t)), Expr.mul(0.5, q)))

        M.solve()

        x_opt = x.level()
        obj_val = M.primalObjValue()
        return obj_val, x_opt, -1


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, mosek_solver)
