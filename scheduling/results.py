from pyomo.environ import value
import pandas as pd
import os


def extract_decision_assignments(model):
    rows = []

    for k in model.K:
        for l in model.L:
            for t in model.T:
                val = value(model.x[k, l, t])
                if val is not None and val > 1e-6:
                    rows.append({
                        "decision": k,
                        "line": l,
                        "day": int(t),
                        "assigned_lbs": float(val)
                    })

    return pd.DataFrame(rows)


def extract_production_vs_demand(model, demand):
    rows = []

    for p in model.P:
        for t in model.T:
            produced = float(value(model.prod[p, t]))
            dem = float(demand.get((p, t), 0.0))

            rows.append({
                "sku": p,
                "day": int(t),
                "produced_lbs": round(produced, 3),
                "demand_lbs": round(dem, 3),
                "gap": round(produced - dem, 3)
            })

    return pd.DataFrame(rows)


def extract_line_load(model, rate, hours_available):
    rows = []

    for l in model.L:
        for t in model.T:
            used = sum(
                value(model.x[k, l, t]) / rate[k]
                for k in model.K
            )

            hrs = hours_available.get((l, t), 0)

            rows.append({
                "line": l,
                "day": int(t),
                "hours_used": round(used, 3),
                "hours_available": round(hrs, 3),
                "util_pct": round(used / hrs * 100, 1) if hrs > 0 else 0
            })

    return pd.DataFrame(rows)


def extract_nonpreferred_usage(model, preferred_matrix):
    rows = []

    for k in model.K:
        for l in model.L:
            for t in model.T:
                val = value(model.x[k, l, t])
                if val is not None and val > 1e-6 and preferred_matrix.get((k, l), 0) == 0:
                    rows.append({
                        "decision": k,
                        "line": l,
                        "day": int(t),
                        "assigned_lbs": float(val),
                        "nonpreferred": 1
                    })

    return pd.DataFrame(rows)


def save_results(df_dict, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    for name, df in df_dict.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


def extract_all_results(model, inputs):
    decision_df = extract_decision_assignments(model)
    prod_vs_dem_df = extract_production_vs_demand(model, inputs["D"])
    line_load_df = extract_line_load(model, inputs["R"], inputs["H"])
    nonpreferred_df = extract_nonpreferred_usage(model, inputs["A"])

    return {
        "decision_assignments": decision_df,
        "production_vs_demand": prod_vs_dem_df,
        "line_load": line_load_df,
        "nonpreferred_usage": nonpreferred_df,
    }