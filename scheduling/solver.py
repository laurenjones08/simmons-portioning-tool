from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus


def solve_model(model, solver_name="highs", tee=False):
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=tee)
    return results


def check_solution(results):
    status = results.solver.status
    term = results.solver.termination_condition

    print(f"Solver status: {status}")
    print(f"Termination condition: {term}")

    if status == SolverStatus.ok and term in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        return True

    if term == TerminationCondition.infeasible:
        raise ValueError("Model is infeasible.")

    raise ValueError(
        f"Solver did not return a usable solution. "
        f"Status={status}, Termination={term}"
    )