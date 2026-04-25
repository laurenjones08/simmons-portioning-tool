from collections import Counter
from typing import Any, Dict, Tuple

from pyomo.environ import Constraint, SolverFactory
from pyomo.opt import TerminationCondition


def _candidate_solver_names(requested_name: str) -> list[str]:
    requested = str(requested_name or "").strip() or "highs"
    if requested == "gurobi":
        # Prefer the APPSI gurobi interface, which is more robust against
        # gurobipy API changes than the older direct interface.
        return ["appsi_gurobi", "gurobi", "appsi_highs", "highs"]
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


def _build_appsi_gurobi_solver():
    try:
        from pyomo.contrib.appsi.solvers.gurobi import Gurobi
    except Exception:
        return None

    solver = Gurobi()
    if not _solver_available(solver):
        return None
    return solver


def _is_appsi_solver(solver: Any) -> bool:
    return hasattr(solver, "config") and hasattr(solver.config, "load_solution")


def _configure_solver(solver: Any, solver_name: str, tee: bool) -> None:
    requested = str(solver_name or "").strip().lower()
    if requested != "gurobi":
        return

    # Ask Gurobi to distinguish infeasible from unbounded instead of
    # collapsing both outcomes into a single presolve status.
    if hasattr(solver, "gurobi_options") and isinstance(getattr(solver, "gurobi_options"), dict):
        solver.gurobi_options["DualReductions"] = 0
        solver.gurobi_options["InfUnbdInfo"] = 1
        if tee:
            solver.gurobi_options["LogToConsole"] = 1
        return

    if hasattr(solver, "options") and isinstance(getattr(solver, "options"), dict):
        solver.options["DualReductions"] = 0
        solver.options["InfUnbdInfo"] = 1


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
        if candidate == "appsi_highs":
            solver = _build_appsi_highs_solver()
        elif candidate == "appsi_gurobi":
            solver = _build_appsi_gurobi_solver()
        else:
            solver = SolverFactory(candidate)
        if solver is None:
            continue
        if not _solver_available(solver):
            continue
        _configure_solver(solver, solver_name=solver_name, tee=tee)
        return _solve_with_solver(solver, model, tee=tee), solver

    attempted_text = ", ".join(attempted)
    raise ValueError(
        "No usable solver backend was available. "
        f"Attempted: {attempted_text}. "
        "The scheduling worker image should provide either Gurobi via `gurobipy` "
        "or the Python-backed HiGHS interface via `highspy`."
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


def _build_gurobi_iis_report(model, solver: Any, limit: int = 25) -> Dict[str, Any] | None:
    solver_model = getattr(solver, "_solver_model", None)
    if solver_model is None:
        return None

    con_map = getattr(solver, "_pyomo_con_to_solver_con_map", None)
    if con_map is None:
        return None

    try:
        solver_model.computeIIS()
    except Exception as exc:
        return {
            "status": "iis_failed",
            "reason": str(exc),
        }

    family_counts: Counter[str] = Counter()
    member_names: list[str] = []

    for con in model.component_data_objects(Constraint, active=True, descend_into=True):
        solver_con = con_map.get(con)
        if solver_con is None:
            continue
        try:
            in_iis = bool(solver_con.getAttr("IISConstr"))
        except Exception:
            continue
        if not in_iis:
            continue

        family = con.parent_component().name
        family_counts[family] += 1
        if len(member_names) < limit:
            member_names.append(con.name)

    if not family_counts:
        return {
            "status": "iis_empty",
            "constraintFamilies": [],
            "constraintMembers": [],
        }

    return {
        "status": "iis_available",
        "constraintFamilies": [
            {"name": name, "count": count}
            for name, count in family_counts.most_common()
        ],
        "constraintMembers": member_names,
    }


def _format_iis_report(report: Dict[str, Any] | None) -> str:
    if not report:
        return ""

    status = str(report.get("status") or "")
    if status == "iis_failed":
        return f" IIS unavailable: {report.get('reason')}."

    families = report.get("constraintFamilies") or []
    members = report.get("constraintMembers") or []
    if not families:
        return " IIS was computed, but no linear constraint members were identified."

    family_text = ", ".join(f"{item['name']} ({item['count']})" for item in families[:5])
    member_text = ", ".join(str(name) for name in members[:8])
    message = f" IIS families: {family_text}."
    if member_text:
        message += f" Example members: {member_text}."
    return message


def check_solution(results, model=None, solver=None):
    status = _solver_status(results)
    term = _termination_condition(results)

    print(f"Solver status: {status}")
    print(f"Termination condition: {term}")

    if _is_termination(term, "optimal", "feasible"):
        return True

    if _is_termination(term, "infeasible"):
        iis_report = None
        if model is not None and solver is not None:
            iis_report = _build_gurobi_iis_report(model, solver)
        raise ValueError(f"Model is infeasible.{_format_iis_report(iis_report)}")

    if _is_termination(term, "infeasibleorunbounded"):
        raise ValueError(
            "Model failed in solver presolve with infeasibleOrUnbounded. "
            "For Gurobi this usually means presolve could not yet distinguish "
            "a true infeasibility from unboundedness. The worker now sets "
            "DualReductions=0 and InfUnbdInfo=1, so rerunning should return a "
            "more specific diagnosis."
        )

    raise ValueError(
        f"Solver did not return a usable solution. "
        f"Status={status}, Termination={term}"
    )
