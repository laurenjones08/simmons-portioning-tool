from typing import Any, Tuple

from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition


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


def _build_appsi_highs_solver():
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs
    except Exception:
        return None

    solver = Highs()
    if not _solver_available(solver):
        return None
    return solver


def _is_appsi_solver(solver: Any) -> bool:
    return hasattr(solver, "config") and hasattr(solver.config, "load_solution")


def _solve_with_solver(solver, model, tee: bool):
    if _is_appsi_solver(solver):
        solver.config.load_solution = False
        if tee and hasattr(solver.config, "stream_solver"):
            solver.config.stream_solver = True
        return solver.solve(model)

    try:
        return solver.solve(model, tee=tee, load_solutions=False)
    except TypeError:
        return solver.solve(model, tee=tee)


def solve_model(model, solver_name="highs", tee=False) -> Tuple[Any, Any]:
    attempted: list[str] = []

    for candidate in _candidate_solver_names(solver_name):
        attempted.append(candidate)
        solver = _build_appsi_highs_solver() if candidate == "appsi_highs" else SolverFactory(candidate)
        if solver is None:
            continue
        if not _solver_available(solver):
            continue
        return _solve_with_solver(solver, model, tee=tee), solver

    attempted_text = ", ".join(attempted)
    raise ValueError(
        "No usable solver backend was available. "
        f"Attempted: {attempted_text}. "
        "The scheduling worker image should provide either the Python-backed "
        "`appsi_highs` solver via `highspy` or the `highs` executable."
    )


def _solver_status(results: Any):
    solver_info = getattr(results, "solver", None)
    return getattr(solver_info, "status", None)


def _termination_condition(results: Any):
    solver_info = getattr(results, "solver", None)
    if solver_info is not None and hasattr(solver_info, "termination_condition"):
        return solver_info.termination_condition
    return getattr(results, "termination_condition", None)


def _termination_name(term: Any) -> str:
    if term is None:
        return ""
    name = getattr(term, "name", None)
    if isinstance(name, str) and name:
        return name.lower()

    text = str(term).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _is_termination(term: Any, *expected: str) -> bool:
    normalized = _termination_name(term)
    return normalized in {value.lower() for value in expected}


def load_solution(model, results: Any, solver: Any) -> None:
    term = _termination_condition(results)
    if not _is_termination(term, "optimal", "feasible"):
        return

    solution_loader = getattr(results, "solution_loader", None)
    if solution_loader is not None and hasattr(solution_loader, "load_vars"):
        solution_loader.load_vars()
        return

    model.solutions.load_from(results)


def check_solution(results):
    status = _solver_status(results)
    term = _termination_condition(results)

    print(f"Solver status: {status}")
    print(f"Termination condition: {term}")

    if _is_termination(term, "optimal", "feasible"):
        return True

    if _is_termination(term, "infeasible"):
        raise ValueError("Model is infeasible.")

    raise ValueError(
        f"Solver did not return a usable solution. "
        f"Status={status}, Termination={term}"
    )
