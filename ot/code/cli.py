import argparse

from util import run_algorithm
from sinkhorn_solver import sinkhorn_algorithm
from gurobi_solver import gurobi_solver
from mosek_solver import mosek_solver


def main():
    parser = argparse.ArgumentParser(description="Run optimal transport algorithms.")
    parser.add_argument("src_path", type=str, help="Path to the source image")
    parser.add_argument("dest_path", type=str, help="Path to the destination image")
    parser.add_argument("algorithm", type=str, choices=["sinkhorn", "gurobi", "mosek"], help="Algorithm to use")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for Sinkhorn algorithm")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max iterations for Sinkhorn algorithm")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for Sinkhorn algorithm")
    parser.add_argument("--method", type=str, default="intpnt",
                        choices=["intpnt", "primalSimplex", "dualSimplex"], help="Method for Mosek solver")

    args = parser.parse_args()

    if args.algorithm == "sinkhorn":
        run_algorithm(args.src_path, args.dest_path, sinkhorn_algorithm, epsilon=args.epsilon, max_iter=args.max_iter,
                      tol=args.tol)
    elif args.algorithm == "gurobi":
        match args.method:
            case "intpnt":
                method = 2
            case "primalSimplex":
                method = 0
            case "dualSimplex":
                method = 1
            case _:
                raise ValueError("Invalid method for Gurobi solver.")
        run_algorithm(args.src_path, args.dest_path, gurobi_solver, method=method)
    elif args.algorithm == "mosek":
        run_algorithm(args.src_path, args.dest_path, mosek_solver, method=args.method)


if __name__ == "__main__":
    main()
