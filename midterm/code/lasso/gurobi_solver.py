from typing import Tuple

import gurobipy as gp
from gurobipy import GRB
from numpy.typing import NDArray

from util import run_algorithm


def gurobi_solver(A: NDArray, b: NDArray, mu: float) -> Tuple[float, NDArray, int]:
    model = gp.Model()

    m, n = A.shape

    x = model.addMVar(n, lb=-GRB.INFINITY, name='x')
    t = model.addMVar(n, lb=0.0, name='t')

    model.addConstr(x <= t, name='constr_upper')
    model.addConstr(-x <= t, name='constr_lower')

    linear_term = mu * t.sum()
    Ax = A @ x
    diff = Ax - b
    quad_term = 0.5 * (diff @ diff)

    model.setObjective(linear_term + quad_term)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_sol = x.X
        obj_val = model.objVal
        iter_num = model.IterCount
        return obj_val, x_sol, iter_num  # type: ignore
    else:
        raise ValueError('Optimization failed')


if __name__ == "__main__":
    run_algorithm(512, 1024, 0.1, 0, 0.01, gurobi_solver)
