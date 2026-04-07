import pandas as pd


def load_short_term_demand(short_term_file, P, T, week1_days, sheet_name=0, start_row=0):
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
    - Missing SKU/day pairs fall back to long-term demand later
    """
    if short_term_file is None:
        return {}

    df = pd.read_excel(short_term_file, sheet_name=sheet_name, header=None)

    if start_row > 0:
        df = df.iloc[start_row:].reset_index(drop=True)

    # 0-based pandas column indices
    # D=3, F=5, I=8, K=10, N=13, P=15, S=18, U=20, X=23, Z=25, AC=28, AE=30
    day_column_map = {
        1: {"sku_col": 3,  "demand_col": 5},   # Monday
        2: {"sku_col": 8,  "demand_col": 10},  # Tuesday
        3: {"sku_col": 13, "demand_col": 15},  # Wednesday
        4: {"sku_col": 18, "demand_col": 20},  # Thursday
        5: {"sku_col": 23, "demand_col": 25},  # Friday
        6: {"sku_col": 28, "demand_col": 30},  # Saturday
    }

    D_short = {}

    for _, row in df.iterrows():
        for day_num, cols in day_column_map.items():
            if day_num not in T or day_num not in week1_days:
                continue

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

            D_short[(sku, day_num)] = D_short.get((sku, day_num), 0.0) + demand

    return D_short


def build_effective_demand(P, T, D_long, D_short, week1_days):
    """
    Week 1 uses short-term demand when available.
    All other days use long-term demand.
    Missing short-term values in week 1 fall back to long-term demand.
    """
    D_eff = {}

    for p in P:
        for t in T:
            if t in week1_days and (p, t) in D_short:
                D_eff[(p, t)] = D_short[(p, t)]
            else:
                D_eff[(p, t)] = D_long[(p, t)]

    return D_eff


def get_model_inputs(short_term_file=None):
    # --- Sets ---
    P = ["A", "B", "C", "D", "E", "F", "G", "H"]  # SKUs
    T = list(range(1, 13))  # Days

    # Decisions are now line-specific
    K = [
        "cut_DSI888",
        "cut_DB20",
        "pack_DSI884",
        "trim_DSI888",
        "trim_DSI884",
        "slice_DSI884",
        "slice_DB20",
        "dice_DB20",
        "grind_DSI888",
    ]

    # Real line names used consistently throughout the file
    L = ["DSI 888", "DSI 884", "DB20"]

    B = [
        "B 0-390", "B 390-440", "B 440-490", "B 490-540",
        "B 540-590", "B 590-640", "B 640-690", "B 690-1000"
    ]  # WIP buckets

    # --- Week logic ---
    # Plant runs 6 days/week (Monday-Saturday)
    # Week 1: days 1-6 use short-term demand when available
    # Week 2: days 7-12 use long-term demand
    week1_days = [1, 2, 3, 4, 5, 6]

    # --- Bucket assigned to each decision ---
    bucket_of_k = {
        "cut_DSI888": "B 0-390",
        "cut_DB20": "B 0-390",
        "pack_DSI884": "B 390-440",
        "trim_DSI888": "B 440-490",
        "trim_DSI884": "B 440-490",
        "slice_DSI884": "B 490-540",
        "slice_DB20": "B 490-540",
        "dice_DB20": "B 540-590",
        "grind_DSI888": "B 590-640",
    }

    # --- Line assigned to each decision ---
    line_of_k = {
        "cut_DSI888": "DSI 888",
        "cut_DB20": "DB20",
        "pack_DSI884": "DSI 884",
        "trim_DSI888": "DSI 888",
        "trim_DSI884": "DSI 884",
        "slice_DSI884": "DSI 884",
        "slice_DB20": "DB20",
        "dice_DB20": "DB20",
        "grind_DSI888": "DSI 888",
    }

    # --- WIP available by bucket and day ---
    WIP = {
        ("B 0-390", 1): 30468.11, ("B 0-390", 2): 30468.11, ("B 0-390", 3): 30468.11, ("B 0-390", 4): 30468.11,
        ("B 0-390", 5): 30468.11, ("B 0-390", 6): 30468.11, ("B 0-390", 7): 30468.11, ("B 0-390", 8): 30468.11,
        ("B 0-390", 9): 30468.11, ("B 0-390", 10): 30468.11, ("B 0-390", 11): 30468.11, ("B 0-390", 12): 30468.11,

        ("B 390-440", 1): 39254.81, ("B 390-440", 2): 39254.81, ("B 390-440", 3): 39254.81, ("B 390-440", 4): 39254.81,
        ("B 390-440", 5): 39254.81, ("B 390-440", 6): 39254.81, ("B 390-440", 7): 39254.81, ("B 390-440", 8): 39254.81,
        ("B 390-440", 9): 39254.81, ("B 390-440", 10): 39254.81, ("B 390-440", 11): 39254.81, ("B 390-440", 12): 39254.81,

        ("B 440-490", 1): 56783.94, ("B 440-490", 2): 56783.94, ("B 440-490", 3): 56783.94, ("B 440-490", 4): 56783.94,
        ("B 440-490", 5): 56783.94, ("B 440-490", 6): 56783.94, ("B 440-490", 7): 56783.94, ("B 440-490", 8): 56783.94,
        ("B 440-490", 9): 56783.94, ("B 440-490", 10): 56783.94, ("B 440-490", 11): 56783.94, ("B 440-490", 12): 56783.94,

        ("B 490-540", 1): 60929.4, ("B 490-540", 2): 60929.4, ("B 490-540", 3): 60929.4, ("B 490-540", 4): 60929.4,
        ("B 490-540", 5): 60929.4, ("B 490-540", 6): 60929.4, ("B 490-540", 7): 60929.4, ("B 490-540", 8): 60929.4,
        ("B 490-540", 9): 60929.4, ("B 490-540", 10): 60929.4, ("B 490-540", 11): 60929.4, ("B 490-540", 12): 60929.4,

        ("B 540-590", 1): 48495.82, ("B 540-590", 2): 48495.82, ("B 540-590", 3): 48495.82, ("B 540-590", 4): 48495.82,
        ("B 540-590", 5): 48495.82, ("B 540-590", 6): 48495.82, ("B 540-590", 7): 48495.82, ("B 540-590", 8): 48495.82,
        ("B 540-590", 9): 48495.82, ("B 540-590", 10): 48495.82, ("B 540-590", 11): 48495.82, ("B 540-590", 12): 48495.82,

        ("B 590-640", 1): 28630.86, ("B 590-640", 2): 28630.86, ("B 590-640", 3): 28630.86, ("B 590-640", 4): 28630.86,
        ("B 590-640", 5): 28630.86, ("B 590-640", 6): 28630.86, ("B 590-640", 7): 28630.86, ("B 590-640", 8): 28630.86,
        ("B 590-640", 9): 28630.86, ("B 590-640", 10): 28630.86, ("B 590-640", 11): 28630.86, ("B 590-640", 12): 28630.86,

        ("B 640-690", 1): 12536.14, ("B 640-690", 2): 12536.14, ("B 640-690", 3): 12536.14, ("B 640-690", 4): 12536.14,
        ("B 640-690", 5): 12536.14, ("B 640-690", 6): 12536.14, ("B 640-690", 7): 12536.14, ("B 640-690", 8): 12536.14,
        ("B 640-690", 9): 12536.14, ("B 640-690", 10): 12536.14, ("B 640-690", 11): 12536.14, ("B 640-690", 12): 12536.14,

        ("B 690-1000", 1): 5250.097, ("B 690-1000", 2): 5250.097, ("B 690-1000", 3): 5250.097, ("B 690-1000", 4): 5250.097,
        ("B 690-1000", 5): 5250.097, ("B 690-1000", 6): 5250.097, ("B 690-1000", 7): 5250.097, ("B 690-1000", 8): 5250.097,
        ("B 690-1000", 9): 5250.097, ("B 690-1000", 10): 5250.097, ("B 690-1000", 11): 5250.097, ("B 690-1000", 12): 5250.097,
    }

    # --- Long-term demand (p, t) ---
    D_long = {
        ("A", 1): 400, ("A", 2): 420, ("A", 3): 440, ("A", 4): 460, ("A", 5): 480, ("A", 6): 500,
        ("A", 7): 520, ("A", 8): 540, ("A", 9): 560, ("A", 10): 580, ("A", 11): 600, ("A", 12): 620,

        ("B", 1): 300, ("B", 2): 315, ("B", 3): 330, ("B", 4): 345, ("B", 5): 360, ("B", 6): 375,
        ("B", 7): 390, ("B", 8): 405, ("B", 9): 420, ("B", 10): 435, ("B", 11): 450, ("B", 12): 465,

        ("C", 1): 350, ("C", 2): 360, ("C", 3): 370, ("C", 4): 380, ("C", 5): 390, ("C", 6): 400,
        ("C", 7): 410, ("C", 8): 420, ("C", 9): 430, ("C", 10): 440, ("C", 11): 450, ("C", 12): 460,

        ("D", 1): 280, ("D", 2): 295, ("D", 3): 310, ("D", 4): 325, ("D", 5): 340, ("D", 6): 355,
        ("D", 7): 370, ("D", 8): 385, ("D", 9): 400, ("D", 10): 415, ("D", 11): 430, ("D", 12): 445,

        ("E", 1): 500, ("E", 2): 510, ("E", 3): 520, ("E", 4): 530, ("E", 5): 540, ("E", 6): 550,
        ("E", 7): 560, ("E", 8): 570, ("E", 9): 580, ("E", 10): 590, ("E", 11): 600, ("E", 12): 610,

        ("F", 1): 260, ("F", 2): 275, ("F", 3): 290, ("F", 4): 305, ("F", 5): 320, ("F", 6): 335,
        ("F", 7): 350, ("F", 8): 365, ("F", 9): 380, ("F", 10): 395, ("F", 11): 410, ("F", 12): 425,

        ("G", 1): 450, ("G", 2): 465, ("G", 3): 480, ("G", 4): 495, ("G", 5): 510, ("G", 6): 525,
        ("G", 7): 540, ("G", 8): 555, ("G", 9): 570, ("G", 10): 585, ("G", 11): 600, ("G", 12): 615,

        ("H", 1): 320, ("H", 2): 330, ("H", 3): 340, ("H", 4): 350, ("H", 5): 360, ("H", 6): 370,
        ("H", 7): 380, ("H", 8): 390, ("H", 9): 400, ("H", 10): 410, ("H", 11): 420, ("H", 12): 430,
    }

    # --- Load short-term demand from weekly Excel file ---
    D_short = load_short_term_demand(
        short_term_file=short_term_file,
        P=P,
        T=T,
        week1_days=week1_days,
        sheet_name=0,
        start_row=0,
    )

    # --- Effective demand used by the model ---
    D_eff = build_effective_demand(
        P=P,
        T=T,
        D_long=D_long,
        D_short=D_short,
        week1_days=week1_days,
    )

    # --- Yield: how much SKU p comes from decision k ---
    Y = {
        ("A", "cut_DSI888"): 0.60, ("A", "cut_DB20"): 0.60, ("A", "pack_DSI884"): 0.40,
        ("A", "trim_DSI888"): 0.55, ("A", "trim_DSI884"): 0.55,
        ("A", "slice_DSI884"): 0.30, ("A", "slice_DB20"): 0.30,
        ("A", "dice_DB20"): 0.20, ("A", "grind_DSI888"): 0.10,

        ("B", "cut_DSI888"): 0.50, ("B", "cut_DB20"): 0.50, ("B", "pack_DSI884"): 0.50,
        ("B", "trim_DSI888"): 0.45, ("B", "trim_DSI884"): 0.45,
        ("B", "slice_DSI884"): 0.35, ("B", "slice_DB20"): 0.35,
        ("B", "dice_DB20"): 0.25, ("B", "grind_DSI888"): 0.15,

        ("C", "cut_DSI888"): 0.40, ("C", "cut_DB20"): 0.40, ("C", "pack_DSI884"): 0.60,
        ("C", "trim_DSI888"): 0.50, ("C", "trim_DSI884"): 0.50,
        ("C", "slice_DSI884"): 0.45, ("C", "slice_DB20"): 0.45,
        ("C", "dice_DB20"): 0.20, ("C", "grind_DSI888"): 0.10,

        ("D", "cut_DSI888"): 0.55, ("D", "cut_DB20"): 0.55, ("D", "pack_DSI884"): 0.35,
        ("D", "trim_DSI888"): 0.60, ("D", "trim_DSI884"): 0.60,
        ("D", "slice_DSI884"): 0.25, ("D", "slice_DB20"): 0.25,
        ("D", "dice_DB20"): 0.15, ("D", "grind_DSI888"): 0.10,

        ("E", "cut_DSI888"): 0.30, ("E", "cut_DB20"): 0.30, ("E", "pack_DSI884"): 0.65,
        ("E", "trim_DSI888"): 0.40, ("E", "trim_DSI884"): 0.40,
        ("E", "slice_DSI884"): 0.50, ("E", "slice_DB20"): 0.50,
        ("E", "dice_DB20"): 0.35, ("E", "grind_DSI888"): 0.20,

        ("F", "cut_DSI888"): 0.45, ("F", "cut_DB20"): 0.45, ("F", "pack_DSI884"): 0.40,
        ("F", "trim_DSI888"): 0.50, ("F", "trim_DSI884"): 0.50,
        ("F", "slice_DSI884"): 0.30, ("F", "slice_DB20"): 0.30,
        ("F", "dice_DB20"): 0.25, ("F", "grind_DSI888"): 0.15,

        ("G", "cut_DSI888"): 0.50, ("G", "cut_DB20"): 0.50, ("G", "pack_DSI884"): 0.55,
        ("G", "trim_DSI888"): 0.45, ("G", "trim_DSI884"): 0.45,
        ("G", "slice_DSI884"): 0.40, ("G", "slice_DB20"): 0.40,
        ("G", "dice_DB20"): 0.30, ("G", "grind_DSI888"): 0.20,

        ("H", "cut_DSI888"): 0.35, ("H", "cut_DB20"): 0.35, ("H", "pack_DSI884"): 0.45,
        ("H", "trim_DSI888"): 0.55, ("H", "trim_DSI884"): 0.55,
        ("H", "slice_DSI884"): 0.50, ("H", "slice_DB20"): 0.50,
        ("H", "dice_DB20"): 0.25, ("H", "grind_DSI888"): 0.15,
    }

    # --- Value per lb of decision ---
    V = {
        "cut_DSI888": 2.0,
        "cut_DB20": 2.0,
        "pack_DSI884": 3.0,
        "trim_DSI888": 2.4,
        "trim_DSI884": 2.4,
        "slice_DSI884": 2.8,
        "slice_DB20": 2.8,
        "dice_DB20": 3.2,
        "grind_DSI888": 1.9,
    }

    # --- Rate (lbs/hour) ---
    R = {
        "cut_DSI888": 100,
        "cut_DB20": 100,
        "pack_DSI884": 80,
        "trim_DSI888": 90,
        "trim_DSI884": 90,
        "slice_DSI884": 85,
        "slice_DB20": 85,
        "dice_DB20": 75,
        "grind_DSI888": 110,
    }

    # --- Hours available per line per day ---
    H = {
        ("DSI 888", 1): 8, ("DSI 888", 2): 8, ("DSI 888", 3): 8, ("DSI 888", 4): 8, ("DSI 888", 5): 8, ("DSI 888", 6): 8,
        ("DSI 888", 7): 8, ("DSI 888", 8): 8, ("DSI 888", 9): 8, ("DSI 888", 10): 8, ("DSI 888", 11): 8, ("DSI 888", 12): 8,

        ("DSI 884", 1): 8, ("DSI 884", 2): 8, ("DSI 884", 3): 8, ("DSI 884", 4): 8, ("DSI 884", 5): 8, ("DSI 884", 6): 8,
        ("DSI 884", 7): 8, ("DSI 884", 8): 8, ("DSI 884", 9): 8, ("DSI 884", 10): 8, ("DSI 884", 11): 8, ("DSI 884", 12): 8,

        ("DB20", 1): 8, ("DB20", 2): 8, ("DB20", 3): 8, ("DB20", 4): 8, ("DB20", 5): 8, ("DB20", 6): 8,
        ("DB20", 7): 8, ("DB20", 8): 8, ("DB20", 9): 8, ("DB20", 10): 8, ("DB20", 11): 8, ("DB20", 12): 8,
    }

    # --- Lag (how many days demand can be delayed) ---
    L_delay = {}

    # --- Line throughput capacity (lbs/hour) ---
    line_throughput = {
        "DSI 888": 9000,
        "DSI 884": 3000,
        "DB20": 67000,
    }

    # --- Objective weights ---
    gamma = 0.1   # demand penalty

    return {
        "P": P,
        "T": T,
        "K": K,
        "L": L,
        "B": B,
        "bucket_of_k": bucket_of_k,
        "line_of_k": line_of_k,
        "WIP": WIP,
        "D_long": D_long,
        "D_short": D_short,
        "D_eff": D_eff,
        "week1_days": week1_days,
        "Y": Y,
        "V": V,
        "R": R,
        "H": H,
        "L_delay": L_delay,
        "line_throughput": line_throughput,
        "gamma": gamma,
    }