"""
A3D3 Two-Stage Solver (Bulk + Cleanup) — DECOMPOSED

LOCKED BUSINESS RULES:
- Pounds - meet required pounds by end of month. Pieces are reported only.
- Two-stage solve:
  * BULK: Class A + B patterns only (Class C forbidden)
  * CLEANUP: Class A + B + C patterns allowed (hard meets remaining pounds)
- Prefer 2 FDS + 1 RTL per input fillet (Class A) via objective nudges; not a hard rule.
- Gram flex ALWAYS stays within [Lower, Upper] for each SKU.
- Capacity: 96 hours/week nominal. If mix needs > nominal hours, we report over-assignment;
  we do NOT fail or under-produce.
- Bulk tries to cover 92% of total required pounds (coverage cap), then Cleanup finishes.

DECOMPOSITION 
- You run 15-20 codes at a time. So we solve in CHUNKS.
- We chunk SKUs by required pounds descending (default chunk_size=20).
- For each chunk: run Bulk then Cleanup using current REMAINING pounds.
- Carry remaining pounds forward to the next chunk.
- Final output is aggregated across chunks for the month.

INPUT
- Excel file with either:
  A) single-month sheet (default: Sheet1) with columns:
     Item Number, Channel (FDS/RTL), Lower Gram Range, Target Gram Size, Upper Gram Range,
     Pounds Required, (optional) Required_Count, Weeks_Month, Pieces_per_Min, Line_Eff
  OR
  B) Month_Config sheet with columns:
     Month, ItemNumber, Channel, LowerGram, TargetGram, UpperGram, RequiredPounds
     (optional) Required_Count, Weeks_Month

OUTPUT
- Excel with:
  Time_Summary
  SKU_Pounds
  Pieces_Report
  Pattern_Execution
  Remaining_Demand
  (optional) Pattern_Library  (can be huge)
"""

import argparse
from itertools import combinations, product
from collections import defaultdict

import pandas as pd
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpInteger, LpStatus, value, PULP_CBC_CMD
)

# ======================
# CONSTANTS
# ======================
FILLET_GRAMS = 479.0
RANDOM_G = 75.0 # grams per random orginal was 75.  Now 95 grams but VBS has a value and SS20 has another value
MAX_RANDOMS = 2
GRAMS_PER_LB = 453.592
PIECES_PER_MIN_DEFAULT = 600
LINE_EFF_DEFAULT = 0.85
NOMINAL_HOURS_PER_WEEK = 96.0

# Objective weights
W_SHORT_LB = 1_000_000
W_OVER_LB  = 5_000
W_TRIM_G   = 1.0
W_RANDOM_G = -10.0
W_CLASS_B  = 100.0
W_CLASS_C  = 5_000.0
W_OVER_HR  = 25_000.0

# Bulk coverage policy
BULK_COVERAGE_TARGET = 0.92

# Bulk preferred window (keeps Bulk from behaving like Cleanup)
PREF_DELTA_FRAC = 0.25

# 3-SKU patterns can explode; we keep full 9-point for 1/2-SKU,
# and use representative points for 3-SKU to keep runtime sane.
TRI_REP_INDEXES = [0, 2, 4, 6, 8]  # uses 5 points out of the 9


# ======================
# FLEX GRID (9-POINT)
# ======================
def gram_grid_9(lower, target, upper):
    """
    9-point grid (1 decimal):
      Lower,
      25/50/75% to Target,
      Target,
      75/50/25% to Upper,
      Upper
    """
    lower = float(lower)
    target = float(target)
    upper = float(upper)

    # clamp target
    if target < lower: target = lower
    if target > upper: target = upper

    pts = []
    # lower -> target
    pts.append(lower)
    pts.append(lower + 0.25 * (target - lower))
    pts.append(lower + 0.50 * (target - lower))
    pts.append(lower + 0.75 * (target - lower))
    pts.append(target)
    # target -> upper
    pts.append(target + 0.75 * (upper - target))
    pts.append(target + 0.50 * (upper - target))
    pts.append(target + 0.25 * (upper - target))
    pts.append(upper)

    # round, unique, in bounds
    out = []
    seen = set()
    for g in pts:
        g1 = round(float(g), 1)
        if lower - 1e-9 <= g1 <= upper + 1e-9 and g1 not in seen:
            out.append(g1)
            seen.add(g1)
    # ensure sorted
    out.sort()
    return out

def preferred_window(lower, target, upper, delta_frac=PREF_DELTA_FRAC):
    lower = float(lower); target = float(target); upper = float(upper)
    rng = max(upper - lower, 0.0)
    d = rng * float(delta_frac)
    lo = max(lower, target - d)
    hi = min(upper, target + d)
    if lo > hi:
        return lower, upper
    return lo, hi

def tri_rep_points(grid9):
    if not grid9:
        return grid9
    out = []
    seen = set()
    for idx in TRI_REP_INDEXES:
        if 0 <= idx < len(grid9):
            g = grid9[idx]
            if g not in seen:
                out.append(g); seen.add(g)
    if not out:
        out = grid9
    return out


# ======================
# PATTERN / CLASSIFICATION
# ======================
def classify_pattern(mix, channel_by_item):
    """
    mix: dict[item -> count] sum(count)==3
    Class A: 2 FDS + 1 RTL
    Class B: other multi-sku mixes
    Class C: single-sku (3 of same)
    """
    if len(mix) == 1:
        return "C"
    fds = 0
    rtl = 0
    for item, cnt in mix.items():
        ch = str(channel_by_item.get(item, "")).strip().upper()
        if ch == "FDS":
            fds += cnt
        elif ch == "RTL":
            rtl += cnt
    if fds == 2 and rtl == 1:
        return "A"
    return "B"

def pattern_throughput_lbs_per_hour(grams_total_portions, pieces_per_min_eff):
    avg_g = float(grams_total_portions) / 3.0
    ppm = float(pieces_per_min_eff)
    return ppm * avg_g / GRAMS_PER_LB * 60.0

def add_pattern(store, mix, grams_by_sku, randoms, channel_by_item, pieces_per_min_eff):
    grams_total = sum(grams_by_sku[k] * v for k, v in mix.items())
    trim = FILLET_GRAMS - grams_total - randoms * RANDOM_G
    if trim < -1e-6:
        return

    key = (
        tuple(sorted(mix.items())),
        tuple(sorted(grams_by_sku.items())),
        int(randoms)
    )
    if key in store:
        return

    cls = classify_pattern(mix, channel_by_item)
    lbph = pattern_throughput_lbs_per_hour(grams_total, pieces_per_min_eff)
    pounds_per_run = grams_total / GRAMS_PER_LB
    hours_per_run = (pounds_per_run / lbph) if lbph > 0 else 0.0

    store[key] = {
        "mix": dict(mix),
        "grams_by_sku": dict(grams_by_sku),
        "grams_total": float(grams_total),
        "randoms": int(randoms),
        "trim": float(trim),
        "class": cls,
        "pounds_per_run": float(pounds_per_run),
        "lb_per_hour": float(lbph),
        "hours_per_run": float(hours_per_run),
    }


# ======================
# INPUT LOADING
# ======================
def _pick(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    raise ValueError(f"Missing column from {cols}")

def load_single_month(df):
    data = pd.DataFrame({
        "item": df[_pick(df, ["Item Number", "ItemNumber"])].astype(str),
        "channel": df[_pick(df, ["Channel", "channel"])].astype(str),
        "lower_g": df[_pick(df, ["Lower Gram Range", "LowerGram", "Lower Gram"])].astype(float),
        "target_g": df[_pick(df, ["Target Gram Size", "TargetGram", "Target Gram"])].astype(float),
        "upper_g": df[_pick(df, ["Upper Gram Range", "UpperGram", "Upper Gram"])].astype(float),
        "required_pounds": df[_pick(df, ["Pounds Required", "RequiredPounds", "Required Pounds"])].fillna(0.0).astype(float),
    })
    if "Required_Count" in df.columns:
        data["required_count"] = df["Required_Count"].fillna(0).astype(int)
    else:
        data["required_count"] = 0

    weeks = None
    ppm = None
    eff = None

    for c in ["Weeks_Month", "WeeksMonth", "Weeks per Month", "WeeksPerMonth"]:
        if c in df.columns:
            try:
                weeks = float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            except Exception:
                weeks = None
            break

    for c in ["Pieces_per_Min", "PiecesPerMin", "Portions_per_Min", "PortionsPerMin", "PPM"]:
        if c in df.columns:
            try:
                ppm = float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            except Exception:
                ppm = None
            break

    for c in ["Line_Eff", "LineEff", "Efficiency", "Eff"]:
        if c in df.columns:
            try:
                eff = float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            except Exception:
                eff = None
            break

    return data, weeks, ppm, eff

def load_month_config(df_mc, default_weeks):
    col_month = _pick(df_mc, ["Month", "month"])
    col_item = _pick(df_mc, ["ItemNumber", "Item Number", "item"])
    col_ch   = _pick(df_mc, ["Channel", "channel"])
    col_lo   = _pick(df_mc, ["LowerGram", "Lower Gram Range", "Lower"])
    col_tg   = _pick(df_mc, ["TargetGram", "Target Gram Size", "Target"])
    col_hi   = _pick(df_mc, ["UpperGram", "Upper Gram Range", "Upper"])
    col_req  = _pick(df_mc, ["RequiredPounds", "Pounds Required", "Required Pounds"])

    col_weeks = None
    for c in ["Weeks_Month", "WeeksMonth", "Weeks per Month", "WeeksPerMonth", "Weeks"]:
        if c in df_mc.columns:
            col_weeks = c
            break

    rows = []
    month_weeks = {}
    for m, g in df_mc.groupby(col_month):
        data = pd.DataFrame({
            "item": g[col_item].astype(str),
            "channel": g[col_ch].astype(str),
            "lower_g": g[col_lo].astype(float),
            "target_g": g[col_tg].astype(float),
            "upper_g": g[col_hi].astype(float),
            "required_pounds": g[col_req].fillna(0.0).astype(float),
            "required_count": g["Required_Count"].fillna(0).astype(int) if "Required_Count" in g.columns else 0,
        })
        data = data[data["required_pounds"] > 0].copy()
        if data.empty:
            continue

        wk = float(default_weeks)
        if col_weeks:
            try:
                wk = float(pd.to_numeric(g[col_weeks], errors="coerce").dropna().iloc[0])
            except Exception:
                wk = float(default_weeks)

        month_weeks[int(m)] = wk
        rows.append((int(m), data))
    return rows, month_weeks


# ======================
# PATTERN GENERATION
# ======================
def generate_patterns(items_df, stage, allow_classes, pieces_per_min, line_eff):
    channel_by_item = dict(zip(items_df["item"], items_df["channel"].astype(str)))
    store = {}

    eff_ppm = float(pieces_per_min) * float(line_eff)
    if eff_ppm <= 0:
        eff_ppm = 1.0

    grids_full = {}
    grids_tri = {}

    for _, r in items_df.iterrows():
        lo, tg, hi = float(r["lower_g"]), float(r["target_g"]), float(r["upper_g"])
        full = gram_grid_9(lo, tg, hi)

        if stage == "Bulk":
            pref_lo, pref_hi = preferred_window(lo, tg, hi)
            pref = [g for g in full if pref_lo - 1e-9 <= g <= pref_hi + 1e-9]
            full = pref if pref else full

        grids_full[r["item"]] = full
        grids_tri[r["item"]] = tri_rep_points(full)

    items = items_df["item"].tolist()

    # Single-SKU (3 portions) => Class C
    for item in items:
        for g in grids_full[item]:
            for rand in range(MAX_RANDOMS + 1):
                if 3*g + rand*RANDOM_G <= FILLET_GRAMS + 1e-9:
                    mix = {item: 3}
                    grams_by = {item: g}
                    cls = classify_pattern(mix, channel_by_item)
                    if cls in allow_classes:
                        add_pattern(store, mix, grams_by, rand, channel_by_item, eff_ppm)

    # Two-SKU (2+1 and 1+2)
    for a, b in combinations(items, 2):
        for ga, gb in product(grids_full[a], grids_full[b]):
            for rand in range(MAX_RANDOMS + 1):
                if 2*ga + gb + rand*RANDOM_G <= FILLET_GRAMS + 1e-9:
                    mix = {a: 2, b: 1}
                    grams_by = {a: ga, b: gb}
                    cls = classify_pattern(mix, channel_by_item)
                    if cls in allow_classes:
                        add_pattern(store, mix, grams_by, rand, channel_by_item, eff_ppm)

                if ga + 2*gb + rand*RANDOM_G <= FILLET_GRAMS + 1e-9:
                    mix = {a: 1, b: 2}
                    grams_by = {a: ga, b: gb}
                    cls = classify_pattern(mix, channel_by_item)
                    if cls in allow_classes:
                        add_pattern(store, mix, grams_by, rand, channel_by_item, eff_ppm)

    # Three-SKU distinct (1+1+1) — bounded grams to keep runtime sane
    for a, b, c in combinations(items, 3):
        for ga, gb, gc in product(grids_tri[a], grids_tri[b], grids_tri[c]):
            for rand in range(MAX_RANDOMS + 1):
                if ga + gb + gc + rand*RANDOM_G <= FILLET_GRAMS + 1e-9:
                    mix = {a: 1, b: 1, c: 1}
                    grams_by = {a: ga, b: gb, c: gc}
                    cls = classify_pattern(mix, channel_by_item)
                    if cls in allow_classes:
                        add_pattern(store, mix, grams_by, rand, channel_by_item, eff_ppm)

    return list(store.values())


# ======================
# SOLVE STAGE (Bulk/Cleanup)
# ======================
def solve_stage(items_df, patterns, stage, nominal_hours, coverage_target=None, time_limit=60, gap=0.002):
    items = items_df["item"].tolist()
    required_lb = dict(zip(items_df["item"], items_df["required_pounds"].astype(float)))
    required_g  = {k: v*GRAMS_PER_LB for k, v in required_lb.items()}

    model = LpProblem(f"A3D3_{stage}", LpMinimize)
    x = {i: LpVariable(f"x_{stage}_{i}", lowBound=0, cat=LpInteger) for i in range(len(patterns))}

    produced_g = {
        item: lpSum(
            x[i] * patterns[i]["mix"].get(item, 0) * patterns[i]["grams_by_sku"].get(item, 0.0)
            for i in x
        )
        for item in items
    }

    hours_used = lpSum(x[i] * patterns[i]["hours_per_run"] for i in x)

    over_hr = LpVariable(f"over_hours_{stage}", lowBound=0)
    model += over_hr >= hours_used - float(nominal_hours)

    short_g = {item: LpVariable(f"short_g_{stage}_{item}", lowBound=0) for item in items}
    over_g  = {item: LpVariable(f"over_g_{stage}_{item}",  lowBound=0) for item in items}

    for item in items:
        model += short_g[item] >= required_g[item] - produced_g[item]
        model += over_g[item]  >= produced_g[item] - required_g[item]

    total_required_g = sum(required_g.values())
    if coverage_target is not None:
        model += lpSum(short_g[item] for item in items) <= (1.0 - float(coverage_target)) * float(total_required_g)
    else:
        for item in items:
            model += produced_g[item] >= required_g[item]

    obj = 0

    # short penalty proportional to required pounds
    for item in items:
        wt = max(required_lb[item], 1.0)
        obj += W_SHORT_LB * (short_g[item] / GRAMS_PER_LB) * wt

    # overproduction penalties
    if stage == "Bulk":
        for item in items:
            over_allow_lb = max(0.005 * required_lb[item], 100.0) if required_lb[item] > 0 else 0.0
            excess_lb = (over_g[item] / GRAMS_PER_LB) - over_allow_lb
            excess_var = LpVariable(f"excess_over_lb_{stage}_{item}", lowBound=0)
            model += excess_var >= excess_lb
            obj += W_OVER_LB * excess_var
    else:
        for item in items:
            obj += (W_OVER_LB * 0.1) * (over_g[item] / GRAMS_PER_LB)

    # pattern trim/random/class penalties
    for i in x:
        p = patterns[i]
        obj += W_TRIM_G * x[i] * p["trim"]
        obj += W_RANDOM_G * x[i] * (p["randoms"] * RANDOM_G)
        if p["class"] == "B":
            obj += W_CLASS_B * x[i]
        elif p["class"] == "C":
            obj += W_CLASS_C * x[i]

    obj += W_OVER_HR * over_hr
    model += obj

    model.solve(PULP_CBC_CMD(timeLimit=int(time_limit), gapRel=float(gap), msg=True))
    status = LpStatus[model.status]

    # Build pattern execution + SKU summaries
    sku_piece = defaultdict(int)
    sku_grams = defaultdict(float)
    pattern_exec = []

    total_fillets = 0
    total_trim_g = 0.0
    total_random_g = 0.0
    total_hours = 0.0
    total_pounds = 0.0

    item_to_channel = dict(zip(items_df["item"], items_df["channel"].astype(str)))

    for i, p in enumerate(patterns):
        qty = int(value(x[i]) or 0)
        if qty <= 0:
            continue

        total_fillets += qty
        total_trim_g += qty * p["trim"]
        total_random_g += qty * p["randoms"] * RANDOM_G
        total_hours += qty * p["hours_per_run"]
        total_pounds += qty * p["pounds_per_run"]

        for item, cnt in p["mix"].items():
            sku_piece[item] += qty * cnt
            sku_grams[item] += qty * cnt * p["grams_by_sku"][item]

        fds_cnt = 0
        rtl_cnt = 0
        for item, cnt in p["mix"].items():
            ch = str(item_to_channel.get(item, "")).upper()
            if ch == "FDS":
                fds_cnt += cnt
            elif ch == "RTL":
                rtl_cnt += cnt

        pattern_exec.append({
            "Stage": stage,
            "Pattern_Class": p["class"],
            "Pattern_ID": i,
            "Fillet_Count": qty,
            "SKU_Mix": dict(p["mix"]),
            "Grams_By_SKU": dict(p["grams_by_sku"]),
            "Portion_Grams_Total": p["grams_total"],
            "Randoms_per_Fillet": p["randoms"],
            "Trim_per_Fillet": p["trim"],
            "Total_Randoms": qty * p["randoms"],
            "Total_Randoms_LB": (qty * p["randoms"] * RANDOM_G) / GRAMS_PER_LB,
            "Total_Trim_Grams": qty * p["trim"],
            "Pounds_per_Run": p["pounds_per_run"],
            "LB_per_Hour": p["lb_per_hour"],
            "Hours_per_Run": p["hours_per_run"],
            "Total_Hours": qty * p["hours_per_run"],
            "FDS_Count": fds_cnt,
            "RTL_Count": rtl_cnt,
        })

    sku_rows = []
    pieces_rows = []
    for _, r in items_df.iterrows():
        item = r["item"]
        grams = float(sku_grams.get(item, 0.0))
        pounds = grams / GRAMS_PER_LB if grams else 0.0
        pieces = int(sku_piece.get(item, 0))

        sku_rows.append({
            "Item": item,
            "Channel": r["channel"],
            "Required_Pounds": float(r["required_pounds"]),
            "Produced_Pounds": pounds,
            "Pound_Gap": pounds - float(r["required_pounds"]),
            "Avg_Gram_per_Portion": (grams / pieces) if pieces else 0.0,
        })

        pieces_rows.append({
            "Item": item,
            "Channel": r["channel"],
            "Required_Count": int(r.get("required_count", 0) or 0),
            "Produced_Count": pieces,
            "Piece_Gap": pieces - int(r.get("required_count", 0) or 0),
        })

    stage_summary = {
        "Stage": stage,
        "Status": status,
        "Objective": float(value(model.objective) or 0.0),
        "Total_Fillets": int(total_fillets),
        "Total_Pounds": float(total_pounds),
        "Total_Randoms_LB": float(total_random_g / GRAMS_PER_LB),
        "Total_Trim_LB": float(total_trim_g / GRAMS_PER_LB),
        "Hours_Used": float(total_hours),
        "Nominal_Hours": float(nominal_hours),
        "Over_Assigned_Hours": max(0.0, float(total_hours - nominal_hours)),
        "Patterns_Considered": int(len(patterns)),
    }

    return status, stage_summary, pd.DataFrame(sku_rows), pd.DataFrame(pieces_rows), pd.DataFrame(pattern_exec)


# ======================
# SOLVE ONE CHUNK (Bulk then Cleanup)
# ======================
def solve_one_chunk(chunk_df, weeks_month, args, pieces_per_min, line_eff):
    chunk_df = chunk_df.copy()
    chunk_df["required_pounds"] = chunk_df["required_pounds"].fillna(0.0).astype(float)
    chunk_df = chunk_df[chunk_df["required_pounds"] > 0].copy()
    if chunk_df.empty:
        return (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame([{"Stage":"Bulk","Status":"NotNeeded"},{"Stage":"Cleanup","Status":"NotNeeded"}])
        )

    nominal_hours = float(weeks_month) * NOMINAL_HOURS_PER_WEEK

    # BULK
    patterns_bulk = generate_patterns(chunk_df, stage="Bulk", allow_classes={"A","B"}, pieces_per_min=pieces_per_min, line_eff=line_eff)

    bulk_status, bulk_summary, bulk_sku, bulk_pieces, bulk_exec = solve_stage(
        chunk_df, patterns_bulk, stage="Bulk", nominal_hours=nominal_hours,
        coverage_target=BULK_COVERAGE_TARGET, time_limit=args.time_limit, gap=args.gap
    )

    produced_lb_bulk = dict(zip(bulk_sku["Item"], bulk_sku["Produced_Pounds"]))
    remaining_rows = []
    for _, r in chunk_df.iterrows():
        item = r["item"]
        req = float(r["required_pounds"])
        prod = float(produced_lb_bulk.get(item, 0.0))
        remaining_rows.append({"item": item, "remaining_lb": max(0.0, req - prod)})
    remaining_df = pd.DataFrame(remaining_rows)

    # CLEANUP on remaining
    cleanup_items = chunk_df.copy()
    cleanup_items["required_pounds"] = cleanup_items["item"].map(
        lambda x: float(remaining_df.loc[remaining_df["item"] == x, "remaining_lb"].iloc[0])
    )
    cleanup_items = cleanup_items[cleanup_items["required_pounds"] > 0].copy()

    if not cleanup_items.empty:
        patterns_cleanup = generate_patterns(cleanup_items, stage="Cleanup", allow_classes={"A","B","C"}, pieces_per_min=pieces_per_min, line_eff=line_eff)
        cleanup_status, cleanup_summary, cleanup_sku, cleanup_pieces, cleanup_exec = solve_stage(
            cleanup_items, patterns_cleanup, stage="Cleanup", nominal_hours=nominal_hours,
            coverage_target=None, time_limit=args.time_limit, gap=args.gap
        )
    else:
        cleanup_status = "NotNeeded"
        cleanup_summary = {
            "Stage": "Cleanup", "Status": "NotNeeded", "Objective": 0.0,
            "Total_Fillets": 0, "Total_Pounds": 0.0, "Total_Randoms_LB": 0.0, "Total_Trim_LB": 0.0,
            "Hours_Used": 0.0, "Nominal_Hours": nominal_hours, "Over_Assigned_Hours": 0.0,
            "Patterns_Considered": 0,
        }
        cleanup_sku = pd.DataFrame(columns=bulk_sku.columns)
        cleanup_pieces = pd.DataFrame(columns=bulk_pieces.columns)
        cleanup_exec = pd.DataFrame(columns=bulk_exec.columns)

    # combine pounds
    bulk_map = dict(zip(bulk_sku["Item"], bulk_sku["Produced_Pounds"])) if not bulk_sku.empty else {}
    cleanup_map = dict(zip(cleanup_sku["Item"], cleanup_sku["Produced_Pounds"])) if not cleanup_sku.empty else {}

    sku_rows = []
    for _, r in chunk_df.iterrows():
        item = r["item"]
        req = float(r["required_pounds"])
        b = float(bulk_map.get(item, 0.0))
        c = float(cleanup_map.get(item, 0.0))
        tot = b + c
        sku_rows.append({
            "Item": item,
            "Channel": r["channel"],
            "Required_Pounds": req,
            "Produced_Pounds": tot,
            "Pound_Gap": tot - req,
            "Driver_Pounds": b,
            "Cleanup_Pounds": c,
            "Cleanup_Pct": (c / tot) if tot > 0 else 0.0,
        })
    sku_pounds = pd.DataFrame(sku_rows)

    # combine pieces
    bulk_p_map = dict(zip(bulk_pieces["Item"], bulk_pieces["Produced_Count"])) if not bulk_pieces.empty else {}
    cleanup_p_map = dict(zip(cleanup_pieces["Item"], cleanup_pieces["Produced_Count"])) if not cleanup_pieces.empty else {}

    pieces_rows = []
    for _, r in chunk_df.iterrows():
        item = r["item"]
        reqc = int(r.get("required_count", 0) or 0)
        prodc = int(bulk_p_map.get(item, 0)) + int(cleanup_p_map.get(item, 0))
        pieces_rows.append({
            "Item": item,
            "Channel": r["channel"],
            "Required_Count": reqc,
            "Produced_Count": prodc,
            "Piece_Gap": prodc - reqc,
        })
    pieces_report = pd.DataFrame(pieces_rows)

    # execution fact
    pattern_exec = pd.concat([bulk_exec, cleanup_exec], ignore_index=True)

    # stage summaries to dataframe
    stage_summ = pd.DataFrame([bulk_summary, cleanup_summary])

    return stage_summ, sku_pounds, pieces_report, pattern_exec, remaining_df


# ======================
# DECOMPOSED MONTH SOLVER (chunks + carry)
# ======================
def chunk_skus_by_pounds(items_df: pd.DataFrame, chunk_size: int):
    df = items_df.sort_values("required_pounds", ascending=False).reset_index(drop=True)
    return [df.iloc[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]

def solve_month_decomposed(items_df, weeks_month, args, pieces_per_min, line_eff, chunk_size):
    # Carry table (remaining pounds live here)
    carry = items_df.copy()
    carry["required_pounds"] = carry["required_pounds"].fillna(0.0).astype(float)
    carry = carry[carry["required_pounds"] > 0].copy()

    # outputs per chunk (we will aggregate later)
    time_all = []
    sku_all = []
    pieces_all = []
    exec_all = []

    chunks = chunk_skus_by_pounds(carry, chunk_size=chunk_size)

    for chunk_idx, chunk in enumerate(chunks, start=1):
        # Build chunk using current remaining pounds
        chunk2 = chunk.copy()
        rem_map = dict(zip(carry["item"], carry["required_pounds"]))
        chunk2["required_pounds"] = chunk2["item"].map(lambda x: float(rem_map.get(x, 0.0)))
        chunk2 = chunk2[chunk2["required_pounds"] > 0].copy()
        if chunk2.empty:
            continue

        stage_summ, sku_pounds, pieces_report, pattern_exec, remaining_df = solve_one_chunk(
            chunk2, weeks_month, args, pieces_per_min, line_eff
        )

        stage_summ.insert(0, "Chunk", chunk_idx)
        sku_pounds.insert(0, "Chunk", chunk_idx)
        pieces_report.insert(0, "Chunk", chunk_idx)
        pattern_exec.insert(0, "Chunk", chunk_idx)

        time_all.append(stage_summ)
        sku_all.append(sku_pounds)
        pieces_all.append(pieces_report)
        exec_all.append(pattern_exec)

        # Update carry: new remaining = remaining_df for chunk items
        rem2 = dict(zip(remaining_df["item"], remaining_df["remaining_lb"]))
        carry.loc[carry["item"].isin(chunk2["item"]), "required_pounds"] = carry["item"].map(lambda x: float(rem2.get(x, 0.0)))

        # prune fully satisfied
        carry = carry[carry["required_pounds"] > 1e-9].copy()

    time_df = pd.concat(time_all, ignore_index=True) if time_all else pd.DataFrame()
    sku_df  = pd.concat(sku_all, ignore_index=True) if sku_all else pd.DataFrame()
    pcs_df  = pd.concat(pieces_all, ignore_index=True) if pieces_all else pd.DataFrame()
    exe_df  = pd.concat(exec_all, ignore_index=True) if exec_all else pd.DataFrame()

    # Aggregate SKU/Pieces across chunks for the final month view
    if not sku_df.empty:
        agg = sku_df.groupby(["Item","Channel"], as_index=False).agg({
            "Required_Pounds":"sum",
            "Produced_Pounds":"sum",
            "Driver_Pounds":"sum",
            "Cleanup_Pounds":"sum",
        })
        agg["Pound_Gap"] = agg["Produced_Pounds"] - agg["Required_Pounds"]
        agg["Cleanup_Pct"] = agg.apply(lambda r: (r["Cleanup_Pounds"]/r["Produced_Pounds"]) if r["Produced_Pounds"]>0 else 0.0, axis=1)
        sku_final = agg
    else:
        sku_final = pd.DataFrame()

    if not pcs_df.empty:
        agg = pcs_df.groupby(["Item","Channel"], as_index=False).agg({
            "Required_Count":"sum",
            "Produced_Count":"sum",
        })
        agg["Piece_Gap"] = agg["Produced_Count"] - agg["Required_Count"]
        pcs_final = agg
    else:
        pcs_final = pd.DataFrame()

    # Remaining demand for audit
    remaining_out = carry[["item","channel","lower_g","target_g","upper_g","required_pounds","required_count"]].copy()
    remaining_out = remaining_out.rename(columns={"required_pounds":"Remaining_Pounds"})
    remaining_out = remaining_out.sort_values("Remaining_Pounds", ascending=False)

    # Time Summary rollup
    nominal_hours = float(weeks_month) * NOMINAL_HOURS_PER_WEEK
    if not time_df.empty:
        # Sum hours used by stage across chunks
        hours_bulk = float(time_df.loc[time_df["Stage"]=="Bulk", "Hours_Used"].sum()) if "Stage" in time_df.columns else 0.0
        hours_cleanup = float(time_df.loc[time_df["Stage"]=="Cleanup", "Hours_Used"].sum()) if "Stage" in time_df.columns else 0.0
        hours_total = hours_bulk + hours_cleanup
        over_total = max(0.0, hours_total - nominal_hours)

        req_total = float(sku_final["Required_Pounds"].sum()) if not sku_final.empty else 0.0
        prod_total = float(sku_final["Produced_Pounds"].sum()) if not sku_final.empty else 0.0
        coverage = 0.0
        if not sku_final.empty and req_total > 0:
            coverage = float((sku_final[["Required_Pounds","Produced_Pounds"]].apply(lambda r: min(r["Required_Pounds"], r["Produced_Pounds"]), axis=1).sum()) / req_total)

        roll = pd.DataFrame([{
            "Chunk":"ALL",
            "Stage":"MONTH_TOTAL",
            "Status":"OK" if remaining_out.empty else "Remaining_Demand",
            "Hours_Used": hours_total,
            "Nominal_Hours": nominal_hours,
            "Over_Assigned_Hours": over_total,
            "Required_Pounds_Total": req_total,
            "Produced_Pounds_Total": prod_total,
            "Coverage_Pct": min(1.0, max(0.0, coverage)),
        }])
        time_out = pd.concat([time_df, roll], ignore_index=True)
    else:
        time_out = pd.DataFrame()

    return time_out, sku_final, pcs_final, exe_df, remaining_out


# ======================
# MAIN
# ======================
def main():
    ap = argparse.ArgumentParser(description="A3D3 Two-Stage Solver (Bulk + Cleanup) — Decomposed")
    ap.add_argument("--input", required=True, help="Input Excel file")
    ap.add_argument("--sheet", default="Sheet1", help="Single-month sheet name (default: Sheet1)")
    ap.add_argument("--output", default="A3D3_two_stage_output.xlsx", help="Output Excel file")
    ap.add_argument("--time_limit", type=int, default=60, help="CBC time limit seconds per stage")
    ap.add_argument("--gap", type=float, default=0.002, help="CBC relative gap")
    ap.add_argument("--weeks_month", type=float, default=4.0, help="Default weeks/month if not found in input")
    ap.add_argument("--pieces_per_min", type=float, default=PIECES_PER_MIN_DEFAULT, help="Nominal portions/min (before efficiency).")
    ap.add_argument("--line_eff", type=float, default=LINE_EFF_DEFAULT, help="Line efficiency multiplier (0-1).")
    ap.add_argument("--use_month_config", action="store_true", help="Solve all months from Month_Config sheet (if present)")
    ap.add_argument("--chunk_size", type=int, default=20, help="Decomposition chunk size (15-20 recommended)")
    ap.add_argument("--write_library", action="store_true", help="Write Pattern_Library sheet (can be huge). Default off.")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.input)

    all_time = []
    all_sku = []
    all_pieces = []
    all_exec = []
    all_remaining = []
    all_lib = []

    if args.use_month_config and "Month_Config" in xls.sheet_names:
        df_mc = pd.read_excel(args.input, sheet_name="Month_Config")
        month_rows, month_weeks = load_month_config(df_mc, default_weeks=args.weeks_month)

        for month, items in month_rows:
            wk = month_weeks.get(month, args.weeks_month)

            # solve decomposed
            time_summary, sku_pounds, pieces_report, pattern_exec, remaining_df = solve_month_decomposed(
                items, wk, args, args.pieces_per_min, args.line_eff, chunk_size=args.chunk_size
            )

            if not time_summary.empty: time_summary.insert(0, "Month", month)
            if not sku_pounds.empty:   sku_pounds.insert(0, "Month", month)
            if not pieces_report.empty: pieces_report.insert(0, "Month", month)
            if not pattern_exec.empty: pattern_exec.insert(0, "Month", month)
            if not remaining_df.empty: remaining_df.insert(0, "Month", month)

            all_time.append(time_summary)
            all_sku.append(sku_pounds)
            all_pieces.append(pieces_report)
            all_exec.append(pattern_exec)
            all_remaining.append(remaining_df)

            # Optional library (warning: can be huge)
            if args.write_library:
                lib = pd.DataFrame(generate_patterns(items, stage="Cleanup", allow_classes={"A","B","C"},
                                                     pieces_per_min=args.pieces_per_min, line_eff=args.line_eff))
                lib.insert(0, "Month", month)
                all_lib.append(lib)

    else:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
        items, weeks_found, ppm_found, eff_found = load_single_month(df)

        weeks = weeks_found if weeks_found is not None else args.weeks_month
        pieces_per_min = ppm_found if ppm_found is not None else args.pieces_per_min
        line_eff = eff_found if eff_found is not None else args.line_eff

        time_summary, sku_pounds, pieces_report, pattern_exec, remaining_df = solve_month_decomposed(
            items, weeks, args, pieces_per_min, line_eff, chunk_size=args.chunk_size
        )

        all_time = [time_summary]
        all_sku = [sku_pounds]
        all_pieces = [pieces_report]
        all_exec = [pattern_exec]
        all_remaining = [remaining_df]

        if args.write_library:
            all_lib = [pd.DataFrame(generate_patterns(items, stage="Cleanup", allow_classes={"A","B","C"},
                                                      pieces_per_min=pieces_per_min, line_eff=line_eff))]

    # Write output
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        pd.concat(all_time, ignore_index=True).to_excel(w, "Time_Summary", index=False)
        pd.concat(all_sku, ignore_index=True).to_excel(w, "SKU_Pounds", index=False)
        pd.concat(all_pieces, ignore_index=True).to_excel(w, "Pieces_Report", index=False)
        pd.concat(all_exec, ignore_index=True).to_excel(w, "Pattern_Execution", index=False)
        pd.concat(all_remaining, ignore_index=True).to_excel(w, "Remaining_Demand", index=False)
        if args.write_library and all_lib:
            pd.concat(all_lib, ignore_index=True).to_excel(w, "Pattern_Library", index=False)

    print(f"[OK] Decomposed two-stage output → {args.output}")


if __name__ == "__main__":
    main()

