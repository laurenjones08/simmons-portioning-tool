from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus


def _candidate_solver_names(requested_name: str) -> list[str]:
    requested = str(requested_name or "").strip() or "highs"
    if requested == "highs":
        # Prefer the Python-backed HiGHS interface from `highspy` when available.
        return ["appsi_highs", "highs"]
    return [requested]


def _solver_available(solver) -> bool:
    available = getattr(solver, "available", None)
    if not callable(available):
        return solver is not None and type(solver).__name__ != "UnknownSolver"

    try:
        return bool(available(exception_flag=False))
    except TypeError:
        try:
            return bool(available())
        except Exception:
            return False
    except Exception:
        return False


def _solve_with_solver(solver, model, tee: bool):
    try:
        return solver.solve(model, tee=tee)
    except TypeError:
        if tee and hasattr(solver, "config") and hasattr(solver.config, "stream_solver"):
            solver.config.stream_solver = True
        return solver.solve(model)


def solve_model(model, solver_name="highs", tee=False):
    attempted: list[str] = []

    for candidate in _candidate_solver_names(solver_name):
        attempted.append(candidate)
        solver = SolverFactory(candidate)
        if not _solver_available(solver):
            continue
        return _solve_with_solver(solver, model, tee=tee)

    attempted_text = ", ".join(attempted)
    raise ValueError(
        "No usable solver backend was available. "
        f"Attempted: {attempted_text}. "
        "The scheduling worker image should provide either the Python-backed "
        "`appsi_highs` solver via `highspy` or the `highs` executable."
    )


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
