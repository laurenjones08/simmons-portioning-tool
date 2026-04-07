from data_prep import get_model_inputs
from model_builder import build_model
from solver import solve_model, check_solution
from results import extract_all_results, save_results


def run_pipeline(short_term_file=None, save_csv=True, output_dir="outputs", tee=False):
    # 1. Load / prepare inputs
    inputs = get_model_inputs(short_term_file=short_term_file)

    # 2. Build model
    model = build_model(
        inputs["P"],
        inputs["T"],
        inputs["K"],
        inputs["L"],
        inputs["B"],
        WIP=inputs["WIP"],
        D_eff=inputs["D_eff"],
        Y=inputs["Y"],
        V=inputs["V"],
        R=inputs["R"],
        H=inputs["H"],
        A=inputs["A"],
        bucket_of_k=inputs["bucket_of_k"],
        L_delay=inputs["L_delay"],
        gamma=inputs["gamma"],
        beta=inputs["beta"],
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