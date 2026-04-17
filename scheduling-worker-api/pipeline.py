import time

from data_prep import get_model_inputs
from model_builder import build_model
from solver import check_solution, load_solution, solve_model
from results import extract_all_results, save_results


def _report_progress(progress_callback, stage, message, details=None):
    if progress_callback is not None:
        progress_callback(stage, message, details or {})


def _debug_mode_enabled(job) -> bool:
    if isinstance(job, dict):
        return bool(job.get("debugMode") or job.get("debug_mode"))
    return bool(getattr(job, "debug_mode", False) or getattr(job, "debugMode", False))


def run_pipeline(
    job=None,
    short_term_file=None,
    save_csv=True,
    output_dir="outputs",
    tee=False,
    plant_id=None,
    sku_ids=None,
    plan_start_date="2026-01-05",
    horizon_days=12,
    progress_callback=None,
):
    timings = {}

    # 1. Load / prepare inputs
    _report_progress(progress_callback, "data_prep", "Preparing model inputs.")
    started = time.perf_counter()
    inputs = get_model_inputs(
        job=job,
        short_term_file=short_term_file,
        plan_start_date=plan_start_date,
        horizon_days=horizon_days,
        plant_id=plant_id,
        sku_ids=sku_ids,
        use_demo_fallbacks=False,
        progress_callback=progress_callback,
    )
    timings["data_prep"] = round(time.perf_counter() - started, 3)
    _report_progress(
        progress_callback,
        "data_prep_complete",
        "Prepared model inputs.",
        {
            "timings": {**inputs.get("dataPrepTimings", {}), "pipeline_data_prep": timings["data_prep"]},
            "counts": inputs.get("dataPrepCounts", {}),
        },
    )
    if _debug_mode_enabled(job):
        _report_progress(
            progress_callback,
            "debug_data_prep_ready",
            "Dataprep snapshot is ready.",
            {"debugDataPrep": inputs},
        )

    # 2. Build model
    _report_progress(progress_callback, "model_build", "Building optimization model.")
    started = time.perf_counter()
    model = build_model(
        inputs["P"],
        inputs["T"],
        inputs["K"],
        inputs["L"],
        inputs["B"],
        inputs["M"],
        WIP=inputs["WIP"],
        D_week1=inputs["D_week1"],
        monthly_contract=inputs["monthly_contract"],
        Y=inputs["Y"],
        V=inputs["V"],
        R=inputs["R"],
        H=inputs["H"],
        bucket_of_k=inputs["bucket_of_k"],
        line_of_k=inputs["line_of_k"],
        month_of_day=inputs["month_of_day"],
        week1_dates=inputs["week1_dates"],
        line_throughput=inputs["line_throughput"],
        big_allowed=inputs["big_allowed"],
        small_allowed=inputs["small_allowed"],
        gamma=inputs["gamma"],
    )
    timings["model_build"] = round(time.perf_counter() - started, 3)
    _report_progress(
        progress_callback,
        "model_build_complete",
        "Built optimization model.",
        {
            "timings": timings,
            "counts": {
                "skuCount": len(inputs["P"]),
                "metricCount": len(inputs["K"]),
                "lineCount": len(inputs["L"]),
                "bucketCount": len(inputs["B"]),
                "dayCount": len(inputs["T"]),
                "monthCount": len(inputs["M"]),
            },
        },
    )
    # 3. Solve
    _report_progress(progress_callback, "solver", "Solving optimization model.")
    started = time.perf_counter()
    solve_results, solver = solve_model(model, solver_name="highs", tee=tee)
    timings["solve"] = round(time.perf_counter() - started, 3)
    _report_progress(
        progress_callback,
        "solver_complete",
        "Solver finished.",
        {
            "timings": timings,
            "solverStatus": str(getattr(getattr(solve_results, "solver", None), "status", None)),
            "terminationCondition": str(
                getattr(
                    getattr(solve_results, "solver", None),
                    "termination_condition",
                    getattr(solve_results, "termination_condition", None),
                )
            ),
        },
    )

    # 4. Check solve status
    check_solution(solve_results)
    load_solution(model, solve_results, solver)

    # 5. Extract outputs
    _report_progress(progress_callback, "extract_results", "Extracting scheduling outputs.")
    started = time.perf_counter()
    output_dfs = extract_all_results(model, inputs)
    timings["extract_results"] = round(time.perf_counter() - started, 3)

    # 6. Save outputs if requested
    if save_csv:
        _report_progress(progress_callback, "save_results", "Saving scheduling CSV outputs.")
        started = time.perf_counter()
        save_results(output_dfs, output_dir=output_dir)
        timings["save_results"] = round(time.perf_counter() - started, 3)

    # 7. Return everything
    timings["total_pipeline"] = round(sum(timings.values()), 3)
    _report_progress(progress_callback, "pipeline_complete", "Scheduling pipeline completed.", {"timings": timings})
    return {
        "inputs": inputs,
        "model": model,
        "solve_results": solve_results,
        "outputs": output_dfs,
        "timings": timings,
    }
