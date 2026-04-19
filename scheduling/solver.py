from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition


def solve_model(model, solver_name="highs", tee=False):
    solver = SolverFactory(solver_name)

    if solver is None or not solver.available(False):
        raise RuntimeError(f"Solver '{solver_name}' is not available.")

    results = solver.solve(model, tee=tee)
    return results


def check_solution(results):
    status = results.solver.status
    termination = results.solver.termination_condition

    if status != SolverStatus.ok:
        raise RuntimeError(
            f"Solver status not OK. Status: {status}, Termination: {termination}"
        )

    if termination not in [TerminationCondition.optimal, TerminationCondition.feasible]:
        raise RuntimeError(
            f"No usable solution found. Status: {status}, Termination: {termination}"
        )