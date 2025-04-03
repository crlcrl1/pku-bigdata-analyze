import gurobipy as gp
import numpy as np
from numpy.typing import NDArray


def gurobi_solver(c: NDArray, alpha: NDArray, beta: NDArray, *, method: int = 0):
    m = len(alpha)
    n = len(beta)
    model = gp.Model("OptimalTransport")
    model.setParam("Method", method)

    pi = model.addVars(m, n, lb=0, name="pi")

    obj = gp.quicksum(c[i, j] * pi[i, j] for i in range(m) for j in range(n))  # type: ignore
    model.setObjective(obj, gp.GRB.MINIMIZE)

    for i in range(m):
        model.addConstr(gp.quicksum(pi[i, j] for j in range(n)) == alpha[i], f"row_{i}")

    for j in range(n):
        model.addConstr(gp.quicksum(pi[i, j] for i in range(m)) == beta[j], f"col_{j}")

    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        solution = np.array([[pi[i, j].X for j in range(n)] for i in range(m)])
        total_cost = model.ObjVal
        return solution, total_cost, -1
    else:
        raise ValueError("Gurobi solver failed")
