import pandas as pd


def load_short_term_demand(short_term_file, P, week1_dates, sheet_name=0, start_row=0):
    """
    Reads short-term demand from a block-formatted Excel file.

    Layout:
        D:H   = Monday
        I:M   = Tuesday
        N:R   = Wednesday
        S:W   = Thursday
        X:AB  = Friday
        AC:AG = Saturday

    Within each day block:
        1st column = SKU
        3rd column = Demand

    Notes:
    - Demand only exists Monday through Saturday
    - Repeated SKU entries within the same day are summed
    - week1_dates must contain 6 production dates in Monday-Saturday order
    """
    if short_term_file is None:
        return {}

    df = pd.read_excel(short_term_file, sheet_name=sheet_name, header=None)

    if start_row > 0:
        df = df.iloc[start_row:].reset_index(drop=True)

    # Map the 6 spreadsheet day blocks to the 6 real dates in week 1
    date_column_map = {
        week1_dates[0]: {"sku_col": 3, "demand_col": 5},    # Monday
        week1_dates[1]: {"sku_col": 8, "demand_col": 10},   # Tuesday
        week1_dates[2]: {"sku_col": 13, "demand_col": 15},  # Wednesday
        week1_dates[3]: {"sku_col": 18, "demand_col": 20},  # Thursday
        week1_dates[4]: {"sku_col": 23, "demand_col": 25},  # Friday
        week1_dates[5]: {"sku_col": 28, "demand_col": 30},  # Saturday
    }

    D_short = {}

    for _, row in df.iterrows():
        for d, cols in date_column_map.items():
            raw_sku = row.iloc[cols["sku_col"]]
            raw_demand = row.iloc[cols["demand_col"]]

            if pd.isna(raw_sku) or pd.isna(raw_demand):
                continue

            sku = str(raw_sku).strip()
            if sku == "" or sku not in P:
                continue

            try:
                demand = float(raw_demand)
            except (TypeError, ValueError):
                continue

            D_short[(sku, d)] = D_short.get((sku, d), 0.0) + demand

    return D_short


def build_week1_demand(P, week1_dates, D_short):
    """
    Builds the week 1 daily demand table.
    Missing SKU/date pairs are set to 0.
    """
    D_week1 = {}
    for p in P:
        for d in week1_dates:
            D_week1[(p, d)] = D_short.get((p, d), 0.0)
    return D_week1


def get_model_inputs(
    job=None,
    short_term_file=None,
    plan_start_date="2026-01-05",
    horizon_days=12,
    plant_id=None,
    sku_ids=None,
    use_demo_fallbacks=False,
):
    """
    plan_start_date:
        First production date in the planning horizon.
        Should be a Monday if you want the short-term file to align naturally.

    horizon_days:
        Number of production days in the horizon, not calendar days.
        Example: 12 means two Mon-Sat production weeks.
    """

    # --- Sets --- SKUs
    P = ["A", "B", "C", "D", "E", "F", "G", "H"]

# mix_metrics API. Pull everything for the skus in the job
    K = [
        "P1_DSI888",
        "P1_DB20",
        "P2_DSI888",
        "P2_DSI884",
        "P3_DSI884",
        "P3_DB20",
        "P4_DB20",
        "P5_DSI888",
        "pack_DSI884",
    ]

# Lines
    L = ["DSI 888", "DSI 884", "DB20"]

#Buckets
    B = [
        "B 0-390", "B 390-440", "B 440-490", "B 490-540",
        "B 540-590", "B 590-640", "B 640-690", "B 690-1000"
    ]

    big_allowed = {
        "A": 1,
        "B": 1,
        "C": 0,
        "D": 1,
        "E": 0,
        "F": 1,
        "G": 1,
        "H": 0,
    }

    small_allowed = {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 0,
        "E": 1,
        "F": 1,
        "G": 0,
        "H": 1,
    }

    bird_type = {}
    for p in P:
        if big_allowed[p] == 1 and small_allowed[p] == 1:
            bird_type[p] = "all"
        elif big_allowed[p] == 1:
            bird_type[p] = "big"
        elif small_allowed[p] == 1:
            bird_type[p] = "small"
        else:
            bird_type[p] = "none"

    # --- Build real production dates (Mon-Sat only) ---
    start_date = pd.Timestamp(plan_start_date)
    all_calendar_days = pd.date_range(start=start_date, periods=40, freq="D")
    production_dates = [d for d in all_calendar_days if d.weekday() < 6][:horizon_days]

    T = production_dates
    week1_dates = T[:6]
    future_dates = T[6:]

    # --- Month logic based on real dates ---
    M = sorted({d.to_period("M") for d in T})
    month_of_day = {d: d.to_period("M") for d in T}

    # --- Monthly contractual demand keyed by month period ---
    # Replace these example values with your real monthly contract data
    monthly_contract = {
        ("A", pd.Period("2026-01", freq="M")): 3000,
        ("A", pd.Period("2026-02", freq="M")): 3200,
        ("B", pd.Period("2026-01", freq="M")): 2200,
        ("B", pd.Period("2026-02", freq="M")): 2400,
        ("C", pd.Period("2026-01", freq="M")): 2600,
        ("C", pd.Period("2026-02", freq="M")): 2700,
        ("D", pd.Period("2026-01", freq="M")): 2100,
        ("D", pd.Period("2026-02", freq="M")): 2300,
        ("E", pd.Period("2026-01", freq="M")): 3400,
        ("E", pd.Period("2026-02", freq="M")): 3500,
        ("F", pd.Period("2026-01", freq="M")): 2000,
        ("F", pd.Period("2026-02", freq="M")): 2150,
        ("G", pd.Period("2026-01", freq="M")): 3100,
        ("G", pd.Period("2026-02", freq="M")): 3250,
        ("H", pd.Period("2026-01", freq="M")): 2500,
        ("H", pd.Period("2026-02", freq="M")): 2600,
    }

    # Keep only months actually present in the horizon
    monthly_contract = {
        (p, m): v
        for (p, m), v in monthly_contract.items()
        if m in M
    }

    # --- Bucket assigned to each decision ---
    #Mix Bucket assignment -> Mix Metric API
    bucket_of_k = {
        "P1_DSI888": "B 0-390",
        "P1_DB20": "B 0-390",
        "P2_DSI888": "B 440-490",
        "P2_DSI884": "B 440-490",
        "P3_DSI884": "B 490-540",
        "P3_DB20": "B 490-540",
        "P4_DB20": "B 540-590",
        "P5_DSI888": "B 590-640",
        "pack_DSI884": "B 390-440",
    }

    # --- Line assigned to each decision ---
    #Mix Line Assignment -> Use MFG Type to assign to a line in the target plant
    line_of_k = {
        "P1_DSI888": "DSI 888",
        "P1_DB20": "DB20",
        "P2_DSI888": "DSI 888",
        "P2_DSI884": "DSI 884",
        "P3_DSI884": "DSI 884",
        "P3_DB20": "DB20",
        "P4_DB20": "DB20",
        "P5_DSI888": "DSI 888",
        "pack_DSI884": "DSI 884",
    }

    # --- Base WIP available by bucket ---
    # Available wip (daily) TODO add SB/BB Deliniation somewhere idk where
    base_wip_by_bucket = {
        "B 0-390": 30468.11,
        "B 390-440": 39254.81,
        "B 440-490": 56783.94,
        "B 490-540": 60929.4,
        "B 540-590": 48495.82,
        "B 590-640": 28630.86,
        "B 640-690": 12536.14,
        "B 690-1000": 5250.097,
    }

    # --- WIP available by bucket and date ---
    WIP = {
        (b, d): base_wip_by_bucket[b]
        for b in B
        for d in T
    }

    # --- Load short-term demand from weekly Excel file ---
    D_short = load_short_term_demand(
        short_term_file=short_term_file,
        P=P,
        week1_dates=week1_dates,
        sheet_name=0,
        start_row=0,
    )

    # --- Week 1 demand only ---
    D_week1 = build_week1_demand(
        P=P,
        week1_dates=week1_dates,
        D_short=D_short,
    )

    # --- Yield: how much SKU p comes from decision k ---
    # Pull from Mix Metric API Unit Plan (SKU, Mix) : % Yield for every mix_metric object
    Y = {
        ("A", "P1_DSI888"): 0.60, ("A", "P1_DB20"): 0.60, ("A", "pack_DSI884"): 0.40,
        ("A", "P2_DSI888"): 0.55, ("A", "P2_DSI884"): 0.55,
        ("A", "P3_DSI884"): 0.30, ("A", "P3_DB20"): 0.30,
        ("A", "P4_DB20"): 0.20, ("A", "P5_DSI888"): 0.10,

        ("B", "P1_DSI888"): 0.50, ("B", "P1_DB20"): 0.50, ("B", "pack_DSI884"): 0.50,
        ("B", "P2_DSI888"): 0.45, ("B", "P2_DSI884"): 0.45,
        ("B", "P3_DSI884"): 0.35, ("B", "P3_DB20"): 0.35,
        ("B", "P4_DB20"): 0.25, ("B", "P5_DSI888"): 0.15,

        ("C", "P1_DSI888"): 0.40, ("C", "P1_DB20"): 0.40, ("C", "pack_DSI884"): 0.60,
        ("C", "P2_DSI888"): 0.50, ("C", "P2_DSI884"): 0.50,
        ("C", "P3_DSI884"): 0.45, ("C", "P3_DB20"): 0.45,
        ("C", "P4_DB20"): 0.20, ("C", "P5_DSI888"): 0.10,

        ("D", "P1_DSI888"): 0.55, ("D", "P1_DB20"): 0.55, ("D", "pack_DSI884"): 0.35,
        ("D", "P2_DSI888"): 0.60, ("D", "P2_DSI884"): 0.60,
        ("D", "P3_DSI884"): 0.25, ("D", "P3_DB20"): 0.25,
        ("D", "P4_DB20"): 0.15, ("D", "P5_DSI888"): 0.10,

        ("E", "P1_DSI888"): 0.30, ("E", "P1_DB20"): 0.30, ("E", "pack_DSI884"): 0.65,
        ("E", "P2_DSI888"): 0.40, ("E", "P2_DSI884"): 0.40,
        ("E", "P3_DSI884"): 0.50, ("E", "P3_DB20"): 0.50,
        ("E", "P4_DB20"): 0.35, ("E", "P5_DSI888"): 0.20,

        ("F", "P1_DSI888"): 0.45, ("F", "P1_DB20"): 0.45, ("F", "pack_DSI884"): 0.40,
        ("F", "P2_DSI888"): 0.50, ("F", "P2_DSI884"): 0.50,
        ("F", "P3_DSI884"): 0.30, ("F", "P3_DB20"): 0.30,
        ("F", "P4_DB20"): 0.25, ("F", "P5_DSI888"): 0.15,

        ("G", "P1_DSI888"): 0.50, ("G", "P1_DB20"): 0.50, ("G", "pack_DSI884"): 0.55,
        ("G", "P2_DSI888"): 0.45, ("G", "P2_DSI884"): 0.45,
        ("G", "P3_DSI884"): 0.40, ("G", "P3_DB20"): 0.40,
        ("G", "P4_DB20"): 0.30, ("G", "P5_DSI888"): 0.20,

        ("H", "P1_DSI888"): 0.35, ("H", "P1_DB20"): 0.35, ("H", "pack_DSI884"): 0.45,
        ("H", "P2_DSI888"): 0.55, ("H", "P2_DSI884"): 0.55,
        ("H", "P3_DSI884"): 0.50, ("H", "P3_DB20"): 0.50,
        ("H", "P4_DB20"): 0.25, ("H", "P5_DSI888"): 0.15,
    }

    # --- Value per lb of decision ---
    # Value for each mix_metric pulled from mix_metric api
    V = {
        "P1_DSI888": 2.0,
        "P1_DB20": 2.0,
        "pack_DSI884": 3.0,
        "P2_DSI888": 2.4,
        "P2_DSI884": 2.4,
        "P3_DSI884": 2.8,
        "P3_DB20": 2.8,
        "P4_DB20": 3.2,
        "P5_DSI888": 1.9,
    }

    # --- Rate (lbs/hour) ---
    # Rate for each mix -> pulled from mix API
    R = {
        "P1_DSI888": 100,
        "P1_DB20": 100,
        "pack_DSI884": 80,
        "P2_DSI888": 90,
        "P2_DSI884": 90,
        "P3_DSI884": 85,
        "P3_DB20": 85,
        "P4_DB20": 75,
        "P5_DSI888": 110,
    }

    # --- Hours available per line per date ---
    # Pulled from Line API
    base_hours_by_line = {
        "DSI 888": 8,
        "DSI 884": 8,
        "DB20": 8,
    }

    H = {
        (l, d): base_hours_by_line[l]
        for l in L
        for d in T
    }

    # --- Lag ---
    L_delay = {}

    # --- Line throughput capacity (lbs/hour) ---
    # Pulled from Lines API
    line_throughput = {
        "DSI 888": 9000,
        "DSI 884": 3000,
        "DB20": 67000,
    }

    # --- Objective weights ---
    gamma = 0.1

    upgrade_pct = {
        "P1_DSI888": 0.08,
        "P1_DB20": 0.07,
        "P2_DSI888": 0.10,
        "P2_DSI884": 0.09,
        "P3_DSI884": 0.12,
        "P3_DB20": 0.11,
        "P4_DB20": 0.15,
        "P5_DSI888": 0.05,
        "pack_DSI884": 0.06,
    }

    return {
        "P": P,
        "T": T,
        "K": K,
        "L": L,
        "B": B,
        "M": M,
        "week1_dates": week1_dates,
        "future_dates": future_dates,
        "month_of_day": month_of_day,
        "bucket_of_k": bucket_of_k,
        "line_of_k": line_of_k,
        "WIP": WIP,
        "D_short": D_short,
        "D_week1": D_week1,
        "monthly_contract": monthly_contract,
        "Y": Y,
        "V": V,
        "R": R,
        "H": H,
        "L_delay": L_delay,
        "line_throughput": line_throughput,
        "gamma": gamma,
        "big_allowed": big_allowed,
        "small_allowed": small_allowed,
        "bird_type": bird_type,
        "upgrade_pct": upgrade_pct,
    }