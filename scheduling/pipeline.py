from data_prep import get_model_inputs
from model_builder import build_model
from solver import solve_model, check_solution
from results import extract_all_results, save_results


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
):
    # 1. Load / prepare inputs
    inputs = get_model_inputs(
        job=job,
        short_term_file=short_term_file,
        plan_start_date=plan_start_date,
        horizon_days=horizon_days,
        plant_id=plant_id,
        sku_ids=sku_ids,
        use_demo_fallbacks=False,
    )

    # 2. Build model
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
    # 3. Solve
    solve_results = solve_model(model, solver_name="highs", tee=tee)

    # 4. Check solve status
    check_solution(solve_results)

    # 5. Extract outputs
    output_dfs = extract_all_results(model, inputs)

    # 6. Save outputs if requested
    if save_csv:
        save_results(output_dfs, output_dir=output_dir)

    # 7. Return everything
    return {
        "inputs": inputs,
        "model": model,
        "solve_results": solve_results,
        "outputs": output_dfs,
    }
