from pyomo.environ import value
import pandas as pd
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 80)


def format_line_name(l):
    s = str(l).strip()

    if s.lower().startswith("line"):
        return "Line" + s[4:]

    return f"Line{s}"


def extract_decision_assignments(model):
    rows = []

    for k in model.K:
        for l in model.L:
            for t in model.T:
                val = value(model.x[k, l, t])
                if val is not None and val > 1e-6:
                    rows.append({
                        "decision": k,
                        "line": format_line_name(l),
                        "shift": int(t),
                        "assigned_lbs": round(float(val), 3)
                    })

    return pd.DataFrame(rows)


def extract_line_schedule(model):
    rows = []

    for t in model.T:
        for l in model.L:
            cuts = []
            for k in model.K:
                val = value(model.x[k, l, t])
                if val is not None and val > 1e-6:
                    cuts.append(f"{k} ({val:.0f} lbs)")

            rows.append({
                "shift": int(t),
                "line": format_line_name(l),
                "cuts": ", ".join(cuts)
            })

    return pd.DataFrame(rows)


def extract_pattern_mix_by_shift(model):
    rows = []

    for t in model.T:
        row = {"shift": int(t)}
        total = 0.0

        for k in model.K:
            amt = sum(value(model.x[k, l, t]) for l in model.L)
            amt = float(amt) if amt is not None else 0.0
            row[str(k)] = round(amt, 2)
            total += amt

        row["TOTAL_LBS"] = round(total, 2)

        for k in model.K:
            row[f"{k}_pct"] = round((row[str(k)] / total) * 100, 1) if total > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_production_vs_demand(model, demand):
    rows = []

    for t in model.T:
        for p in model.P:
            produced = value(model.prod[p, t])
            produced = float(produced) if produced is not None else 0.0
            dem = float(demand.get((p, t), 0.0))

            rows.append({
                "shift": int(t),
                "sku": p,
                "produced_lbs": round(produced, 3),
                "demand_lbs": round(dem, 3),
                "over_under": round(produced - dem, 3)
            })

    return pd.DataFrame(rows)


def extract_line_load(model, rate, hours_available):
    rows = []

    for t in model.T:
        for l in model.L:
            total_input = sum(
                value(model.x[k, l, t]) for k in model.K
            )
            total_input = float(total_input) if total_input is not None else 0.0

            hours_used = sum(
                value(model.x[k, l, t]) / rate[k]
                for k in model.K
                if rate[k] > 0 and value(model.x[k, l, t]) is not None
            )

            hrs = float(hours_available.get((l, t), 0.0))

            rows.append({
                "shift": int(t),
                "line": format_line_name(l),
                "total_input_lbs": round(total_input, 2),
                "hours_used": round(hours_used, 2),
                "hours_available": round(hrs, 2),
                "util_pct": round(hours_used / hrs * 100, 1) if hrs > 0 else 0.0
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
                        "line": format_line_name(l),
                        "shift": int(t),
                        "assigned_lbs": round(float(val), 3),
                        "nonpreferred": 1
                    })

    return pd.DataFrame(rows)


def save_results(df_dict, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    for name, df in df_dict.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Wrote: {path}")
        print(df.head(20))
        print()


def extract_all_results(model, inputs):
    decision_df = extract_decision_assignments(model)
    line_schedule_df = extract_line_schedule(model)
    pattern_mix_df = extract_pattern_mix_by_shift(model)
    prod_vs_dem_df = extract_production_vs_demand(model, inputs["D"])
    line_load_df = extract_line_load(model, inputs["R"], inputs["H"])
    nonpreferred_df = extract_nonpreferred_usage(model, inputs["A"])

    return {
        "x_long_nonzero": decision_df,
        "line_schedule": line_schedule_df,
        "pattern_mix_by_shift": pattern_mix_df,
        "line_load_by_shift": line_load_df,
        "production_vs_demand_by_shift": prod_vs_dem_df,
        "nonpreferred_usage": nonpreferred_df,
    }