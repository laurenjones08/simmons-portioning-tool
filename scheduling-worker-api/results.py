from numpy_compat import ensure_numpy_legacy_aliases

ensure_numpy_legacy_aliases()

from pyomo.environ import value
import pandas as pd
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 80)


def format_line_name(l):
    return str(l).strip()


def format_date(d):
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def format_month(m):
    return str(m)


def extract_decision_assignments(model, bucket_of_k, line_of_k):
    rows = []

    for k in model.K:
        for t in model.T:
            val = value(model.x[k, t])
            if val is not None and val > 1e-6:
                rows.append({
                    "decision": str(k),
                    "bucket": bucket_of_k[str(k)] if str(k) in bucket_of_k else bucket_of_k[k],
                    "line": format_line_name(line_of_k[str(k)] if str(k) in line_of_k else line_of_k[k]),
                    "date": format_date(t),
                    "assigned_lbs": round(float(val), 3)
                })

    return pd.DataFrame(rows)


def extract_line_schedule(model, line_of_k):
    rows = []

    for t in model.T:
        for l in model.L:
            cuts = []
            for k in model.K:
                line_name = line_of_k[str(k)] if str(k) in line_of_k else line_of_k[k]
                if line_name != l:
                    continue

                val = value(model.x[k, t])
                if val is not None and val > 1e-6:
                    cuts.append(f"{k} ({val:.0f} lbs)")

            rows.append({
                "date": format_date(t),
                "line": format_line_name(l),
                "cuts": ", ".join(cuts)
            })

    return pd.DataFrame(rows)


def extract_pattern_mix_by_date(model):
    rows = []

    for t in model.T:
        row = {"date": format_date(t)}
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


def extract_production_vs_demand(model, D_short, D_week1, week1_dates, bird_type):
    rows = []

    week1_date_set = set(week1_dates)

    for t in model.T:
        for p in model.P:
            produced = value(model.prod[p, t])
            produced = float(produced) if produced is not None else 0.0

            short_dem = float(D_short.get((str(p), t), 0.0))
            week1_dem = float(D_week1.get((str(p), t), 0.0))

            rows.append({
                "date": format_date(t),
                "sku": str(p),
                "bird_type": bird_type[str(p)] if str(p) in bird_type else bird_type[p],
                "produced_lbs": round(produced, 3),
                "short_term_demand_lbs": round(short_dem, 3) if t in week1_date_set else 0.0,
                "week1_demand_lbs": round(week1_dem, 3) if t in week1_date_set else 0.0,
                "gap_to_short": round(produced - short_dem, 3) if t in week1_date_set else None,
                "gap_to_week1": round(produced - week1_dem, 3) if t in week1_date_set else None,
            })

    return pd.DataFrame(rows)


def extract_monthly_contract_summary(model, monthly_contract, month_of_day):
    rows = []

    for p in model.P:
        for m in model.M:
            produced = sum(
                value(model.prod[p, t])
                for t in model.T
                if month_of_day[t] == m and value(model.prod[p, t]) is not None
            )
            produced = float(produced) if produced is not None else 0.0

            contract = float(monthly_contract.get((str(p), m), 0.0))

            rows.append({
                "sku": str(p),
                "month": format_month(m),
                "produced_lbs": round(produced, 3),
                "monthly_contract_lbs": round(contract, 3),
                "gap_to_contract": round(produced - contract, 3),
            })

    return pd.DataFrame(rows)


def extract_line_load(model, rate, hours_available, line_of_k, line_throughput):
    rows = []

    for t in model.T:
        for l in model.L:
            total_input = sum(
                value(model.x[k, t])
                for k in model.K
                if (line_of_k[str(k)] if str(k) in line_of_k else line_of_k[k]) == l
                and value(model.x[k, t]) is not None
            )
            total_input = float(total_input) if total_input is not None else 0.0

            hours_used = sum(
                value(model.x[k, t]) / rate[str(k)] if str(k) in rate else value(model.x[k, t]) / rate[k]
                for k in model.K
                if (line_of_k[str(k)] if str(k) in line_of_k else line_of_k[k]) == l
                and (rate[str(k)] if str(k) in rate else rate[k]) > 0
                and value(model.x[k, t]) is not None
            )

            hrs = float(hours_available.get((str(l), t), hours_available.get((l, t), 0.0)))
            throughput_cap_lbs = float(line_throughput.get(str(l), line_throughput.get(l, 0.0))) * hrs

            rows.append({
                "date": format_date(t),
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
                if (bucket_of_k[str(k)] if str(k) in bucket_of_k else bucket_of_k[k]) == b
                and value(model.x[k, t]) is not None
            )
            used = float(used) if used is not None else 0.0
            avail = float(wip.get((str(b), t), wip.get((b, t), 0.0)))

            rows.append({
                "date": format_date(t),
                "bucket": str(b),
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

    pattern_mix_df = extract_pattern_mix_by_date(model)

    prod_vs_dem_df = extract_production_vs_demand(
        model=model,
        D_short=inputs["D_short"],
        D_week1=inputs["D_week1"],
        week1_dates=inputs["week1_dates"],
        bird_type=inputs["bird_type"],
    )

    monthly_contract_df = extract_monthly_contract_summary(
        model=model,
        monthly_contract=inputs["monthly_contract"],
        month_of_day=inputs["month_of_day"],
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
        "pattern_mix_by_date": pattern_mix_df,
        "line_load_by_date": line_load_df,
        "production_vs_demand_by_date": prod_vs_dem_df,
        "monthly_contract_summary": monthly_contract_df,
        "bucket_usage_by_date": bucket_usage_df,
    }
