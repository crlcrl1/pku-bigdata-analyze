from util import run_algorithm

from mosek.fusion import Model, Domain, Expr, ObjectiveSense
from numpy.typing import NDArray


def mosek_solver(A: NDArray, b: NDArray, mu: float) -> tuple[float, NDArray, int]:
    m, n = A.shape
    with Model("LASSO") as model:
        x = model.variable("x", n, Domain.unbounded())

        t = model.variable("t", n, Domain.greaterThan(0.0))

        model.constraint(Expr.sub(t, x), Domain.greaterThan(0.0))
        model.constraint(Expr.add(t, x), Domain.greaterThan(0.0))

        r = model.variable("r", m, Domain.unbounded())
        model.constraint(Expr.sub(Expr.mul(A, x), r), Domain.equalsTo(b))

        q = model.variable("q", 1, Domain.greaterThan(0.0))

        model.constraint(Expr.vstack(q, r), Domain.inQCone())

        model.objective(ObjectiveSense.Minimize, Expr.add(Expr.mul(mu, Expr.sum(t)), Expr.mul(0.5, q)))

        model.solve()

        x_opt = x.level()
        obj_val = model.primalObjValue()
        return obj_val, x_opt, -1


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, mosek_solver)
