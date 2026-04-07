from pyomo.environ import value
import pandas as pd
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 80)


def format_line_name(l):
    return str(l).strip()


def extract_decision_assignments(model, bucket_of_k, line_of_k):
    rows = []

    for k in model.K:
        for t in model.T:
            val = value(model.x[k, t])
            if val is not None and val > 1e-6:
                rows.append({
                    "decision": k,
                    "bucket": bucket_of_k[k],
                    "line": format_line_name(line_of_k[k]),
                    "shift": int(t),
                    "assigned_lbs": round(float(val), 3)
                })

    return pd.DataFrame(rows)


def extract_line_schedule(model, line_of_k):
    rows = []

    for t in model.T:
        for l in model.L:
            cuts = []
            for k in model.K:
                if line_of_k[k] != l:
                    continue

                val = value(model.x[k, t])
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
            amt = value(model.x[k, t])
            amt = float(amt) if amt is not None else 0.0
            row[str(k)] = round(amt, 2)
            total += amt

        row["TOTAL_LBS"] = round(total, 2)

        for k in model.K:
            row[f"{k}_pct"] = round((row[str(k)] / total) * 100, 1) if total > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_production_vs_demand(model, D_short, D_long, D_eff, week1_days):
    rows = []

    for t in model.T:
        for p in model.P:
            produced = value(model.prod[p, t])
            produced = float(produced) if produced is not None else 0.0

            short_dem = float(D_short.get((p, t), 0.0)) if t in week1_days else 0.0
            long_dem = float(D_long.get((p, t), 0.0))
            eff_dem = float(D_eff.get((p, t), 0.0))

            rows.append({
                "shift": int(t),
                "sku": p,
                "produced_lbs": round(produced, 3),
                "short_term_demand_lbs": round(short_dem, 3),
                "long_term_demand_lbs": round(long_dem, 3),
                "effective_demand_lbs": round(eff_dem, 3),
                "gap_to_short": round(produced - short_dem, 3) if t in week1_days else None,
                "gap_to_long": round(produced - long_dem, 3),
                "gap_to_effective": round(produced - eff_dem, 3),
            })

    return pd.DataFrame(rows)


def extract_line_load(model, rate, hours_available, line_of_k, line_throughput):
    rows = []

    for t in model.T:
        for l in model.L:
            total_input = sum(
                value(model.x[k, t])
                for k in model.K
                if line_of_k[k] == l and value(model.x[k, t]) is not None
            )
            total_input = float(total_input) if total_input is not None else 0.0

            hours_used = sum(
                value(model.x[k, t]) / rate[k]
                for k in model.K
                if line_of_k[k] == l and rate[k] > 0 and value(model.x[k, t]) is not None
            )

            hrs = float(hours_available.get((l, t), 0.0))
            throughput_cap_lbs = float(line_throughput.get(l, 0.0)) * hrs

            rows.append({
                "shift": int(t),
                "line": format_line_name(l),
                "total_input_lbs": round(total_input, 2),
                "hours_used": round(hours_used, 2),
                "hours_available": round(hrs, 2),
                "util_pct": round(hours_used / hrs * 100, 1) if hrs > 0 else 0.0,
                "throughput_cap_lbs": round(throughput_cap_lbs, 2),
                "throughput_util_pct": round(total_input / throughput_cap_lbs * 100, 1) if throughput_cap_lbs > 0 else 0.0,
            })

    return pd.DataFrame(rows)


def extract_bucket_usage(model, bucket_of_k, wip):
    rows = []

    for t in model.T:
        for b in model.B:
            used = sum(
                value(model.x[k, t])
                for k in model.K
                if bucket_of_k[k] == b and value(model.x[k, t]) is not None
            )
            used = float(used) if used is not None else 0.0
            avail = float(wip.get((b, t), 0.0))

            rows.append({
                "shift": int(t),
                "bucket": b,
                "used_lbs": round(used, 2),
                "available_lbs": round(avail, 2),
                "remaining_lbs": round(avail - used, 2),
                "util_pct": round(used / avail * 100, 1) if avail > 0 else 0.0
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
    decision_df = extract_decision_assignments(
        model,
        inputs["bucket_of_k"],
        inputs["line_of_k"]
    )

    line_schedule_df = extract_line_schedule(
        model,
        inputs["line_of_k"]
    )

    pattern_mix_df = extract_pattern_mix_by_shift(model)

    prod_vs_dem_df = extract_production_vs_demand(
        model=model,
        D_short=inputs["D_short"],
        D_long=inputs["D_long"],
        D_eff=inputs["D_eff"],
        week1_days=inputs["week1_days"],
    )

    line_load_df = extract_line_load(
        model,
        inputs["R"],
        inputs["H"],
        inputs["line_of_k"],
        inputs["line_throughput"]
    )

    bucket_usage_df = extract_bucket_usage(
        model,
        inputs["bucket_of_k"],
        inputs["WIP"]
    )

    return {
        "x_long_nonzero": decision_df,
        "line_schedule": line_schedule_df,
        "pattern_mix_by_shift": pattern_mix_df,
        "line_load_by_shift": line_load_df,
        "production_vs_demand_by_shift": prod_vs_dem_df,
        "bucket_usage_by_shift": bucket_usage_df,
    }