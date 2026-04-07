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
    P = ["A", "B", "C", "D", "E", "F", "G", "H"]           # SKUs
    T = list(range(1, 15))                                 # Days
    K = ["cut", "pack", "trim", "slice", "dice", "grind"] # Decisions
    L = ["line1", "line2", "line3", "line4"]              # Lines
    B = ["bucket1", "bucket2", "bucket3"]                 # WIP buckets

    # --- Week logic ---
    # Demand only occurs Monday-Saturday in the short-term file
    # So week 1 is modeled as days 1-6 for short-term demand
    # Days 7-14 fall back to long-term demand
    week1_days = [1, 2, 3, 4, 5, 6]

    # --- Bucket assigned to each decision ---
    bucket_of_k = {
        "cut": "bucket1",
        "pack": "bucket2",
        "trim": "bucket1",
        "slice": "bucket2",
        "dice": "bucket3",
        "grind": "bucket3",
    }

    # --- WIP available by bucket and day ---
    WIP = {
        ("bucket1", 1): 1200, ("bucket1", 2): 1250, ("bucket1", 3): 1300, ("bucket1", 4): 1280,
        ("bucket1", 5): 1350, ("bucket1", 6): 1400, ("bucket1", 7): 1380, ("bucket1", 8): 1450,
        ("bucket1", 9): 1420, ("bucket1", 10): 1500, ("bucket1", 11): 1480, ("bucket1", 12): 1550,
        ("bucket1", 13): 1520, ("bucket1", 14): 1600,

        ("bucket2", 1): 1000, ("bucket2", 2): 1025, ("bucket2", 3): 1050, ("bucket2", 4): 1075,
        ("bucket2", 5): 1100, ("bucket2", 6): 1125, ("bucket2", 7): 1150, ("bucket2", 8): 1175,
        ("bucket2", 9): 1200, ("bucket2", 10): 1225, ("bucket2", 11): 1250, ("bucket2", 12): 1275,
        ("bucket2", 13): 1300, ("bucket2", 14): 1325,

        ("bucket3", 1): 900, ("bucket3", 2): 940, ("bucket3", 3): 980, ("bucket3", 4): 1020,
        ("bucket3", 5): 1060, ("bucket3", 6): 1100, ("bucket3", 7): 1080, ("bucket3", 8): 1120,
        ("bucket3", 9): 1160, ("bucket3", 10): 1200, ("bucket3", 11): 1180, ("bucket3", 12): 1220,
        ("bucket3", 13): 1260, ("bucket3", 14): 1300,
    }

    # --- Long-term demand (p, t) ---
    D_long = {
        ("A", 1): 400, ("A", 2): 420, ("A", 3): 440, ("A", 4): 460, ("A", 5): 480, ("A", 6): 500, ("A", 7): 520,
        ("A", 8): 540, ("A", 9): 560, ("A", 10): 580, ("A", 11): 600, ("A", 12): 620, ("A", 13): 640, ("A", 14): 660,

        ("B", 1): 300, ("B", 2): 315, ("B", 3): 330, ("B", 4): 345, ("B", 5): 360, ("B", 6): 375, ("B", 7): 390,
        ("B", 8): 405, ("B", 9): 420, ("B", 10): 435, ("B", 11): 450, ("B", 12): 465, ("B", 13): 480, ("B", 14): 495,

        ("C", 1): 350, ("C", 2): 360, ("C", 3): 370, ("C", 4): 380, ("C", 5): 390, ("C", 6): 400, ("C", 7): 410,
        ("C", 8): 420, ("C", 9): 430, ("C", 10): 440, ("C", 11): 450, ("C", 12): 460, ("C", 13): 470, ("C", 14): 480,

        ("D", 1): 280, ("D", 2): 295, ("D", 3): 310, ("D", 4): 325, ("D", 5): 340, ("D", 6): 355, ("D", 7): 370,
        ("D", 8): 385, ("D", 9): 400, ("D", 10): 415, ("D", 11): 430, ("D", 12): 445, ("D", 13): 460, ("D", 14): 475,

        ("E", 1): 500, ("E", 2): 510, ("E", 3): 520, ("E", 4): 530, ("E", 5): 540, ("E", 6): 550, ("E", 7): 560,
        ("E", 8): 570, ("E", 9): 580, ("E", 10): 590, ("E", 11): 600, ("E", 12): 610, ("E", 13): 620, ("E", 14): 630,

        ("F", 1): 260, ("F", 2): 275, ("F", 3): 290, ("F", 4): 305, ("F", 5): 320, ("F", 6): 335, ("F", 7): 350,
        ("F", 8): 365, ("F", 9): 380, ("F", 10): 395, ("F", 11): 410, ("F", 12): 425, ("F", 13): 440, ("F", 14): 455,

        ("G", 1): 450, ("G", 2): 465, ("G", 3): 480, ("G", 4): 495, ("G", 5): 510, ("G", 6): 525, ("G", 7): 540,
        ("G", 8): 555, ("G", 9): 570, ("G", 10): 585, ("G", 11): 600, ("G", 12): 615, ("G", 13): 630, ("G", 14): 645,

        ("H", 1): 320, ("H", 2): 330, ("H", 3): 340, ("H", 4): 350, ("H", 5): 360, ("H", 6): 370, ("H", 7): 380,
        ("H", 8): 390, ("H", 9): 400, ("H", 10): 410, ("H", 11): 420, ("H", 12): 430, ("H", 13): 440, ("H", 14): 450,
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
        ("A", "cut"): 0.60, ("A", "pack"): 0.40, ("A", "trim"): 0.55, ("A", "slice"): 0.30, ("A", "dice"): 0.20, ("A", "grind"): 0.10,
        ("B", "cut"): 0.50, ("B", "pack"): 0.50, ("B", "trim"): 0.45, ("B", "slice"): 0.35, ("B", "dice"): 0.25, ("B", "grind"): 0.15,
        ("C", "cut"): 0.40, ("C", "pack"): 0.60, ("C", "trim"): 0.50, ("C", "slice"): 0.45, ("C", "dice"): 0.20, ("C", "grind"): 0.10,
        ("D", "cut"): 0.55, ("D", "pack"): 0.35, ("D", "trim"): 0.60, ("D", "slice"): 0.25, ("D", "dice"): 0.15, ("D", "grind"): 0.10,
        ("E", "cut"): 0.30, ("E", "pack"): 0.65, ("E", "trim"): 0.40, ("E", "slice"): 0.50, ("E", "dice"): 0.35, ("E", "grind"): 0.20,
        ("F", "cut"): 0.45, ("F", "pack"): 0.40, ("F", "trim"): 0.50, ("F", "slice"): 0.30, ("F", "dice"): 0.25, ("F", "grind"): 0.15,
        ("G", "cut"): 0.50, ("G", "pack"): 0.55, ("G", "trim"): 0.45, ("G", "slice"): 0.40, ("G", "dice"): 0.30, ("G", "grind"): 0.20,
        ("H", "cut"): 0.35, ("H", "pack"): 0.45, ("H", "trim"): 0.55, ("H", "slice"): 0.50, ("H", "dice"): 0.25, ("H", "grind"): 0.15,
    }

    # --- Value per lb of decision ---
    V = {
        "cut": 2.0,
        "pack": 3.0,
        "trim": 2.4,
        "slice": 2.8,
        "dice": 3.2,
        "grind": 1.9,
    }

    # --- Rate (lbs/hour) ---
    R = {
        "cut": 100,
        "pack": 80,
        "trim": 90,
        "slice": 85,
        "dice": 75,
        "grind": 110,
    }

    # --- Hours available per line per day ---
    H = {
        ("line1", 1): 8, ("line1", 2): 8, ("line1", 3): 8, ("line1", 4): 8, ("line1", 5): 8, ("line1", 6): 8, ("line1", 7): 8,
        ("line1", 8): 8, ("line1", 9): 8, ("line1", 10): 8, ("line1", 11): 8, ("line1", 12): 8, ("line1", 13): 8, ("line1", 14): 8,

        ("line2", 1): 7, ("line2", 2): 7, ("line2", 3): 7, ("line2", 4): 7, ("line2", 5): 7, ("line2", 6): 7, ("line2", 7): 7,
        ("line2", 8): 7, ("line2", 9): 7, ("line2", 10): 7, ("line2", 11): 7, ("line2", 12): 7, ("line2", 13): 7, ("line2", 14): 7,

        ("line3", 1): 9, ("line3", 2): 9, ("line3", 3): 9, ("line3", 4): 9, ("line3", 5): 9, ("line3", 6): 9, ("line3", 7): 9,
        ("line3", 8): 9, ("line3", 9): 9, ("line3", 10): 9, ("line3", 11): 9, ("line3", 12): 9, ("line3", 13): 9, ("line3", 14): 9,

        ("line4", 1): 6, ("line4", 2): 6, ("line4", 3): 6, ("line4", 4): 6, ("line4", 5): 6, ("line4", 6): 6, ("line4", 7): 6,
        ("line4", 8): 6, ("line4", 9): 6, ("line4", 10): 6, ("line4", 11): 6, ("line4", 12): 6, ("line4", 13): 6, ("line4", 14): 6,
    }

    # --- Preferred line matrix A[k,l] ---
    A = {
        ("cut", "line1"): 1,
        ("cut", "line2"): 0,
        ("cut", "line3"): 1,
        ("cut", "line4"): 0,

        ("pack", "line1"): 0,
        ("pack", "line2"): 1,
        ("pack", "line3"): 0,
        ("pack", "line4"): 1,

        ("trim", "line1"): 1,
        ("trim", "line2"): 1,
        ("trim", "line3"): 0,
        ("trim", "line4"): 0,

        ("slice", "line1"): 0,
        ("slice", "line2"): 1,
        ("slice", "line3"): 1,
        ("slice", "line4"): 0,

        ("dice", "line1"): 0,
        ("dice", "line2"): 0,
        ("dice", "line3"): 1,
        ("dice", "line4"): 1,

        ("grind", "line1"): 1,
        ("grind", "line2"): 0,
        ("grind", "line3"): 0,
        ("grind", "line4"): 1,
    }

    # --- Lag (how many days demand can be delayed) ---
    L_delay = {
        "A": 0,
        "B": 0,
        "C": 1,
        "D": 1,
        "E": 2,
        "F": 2,
        "G": 1,
        "H": 0,
    }

    # --- Objective weights ---
    gamma = 0.1   # demand penalty
    beta = 5.0    # penalty for non-preferred line use

    return {
        "P": P,
        "T": T,
        "K": K,
        "L": L,
        "B": B,
        "bucket_of_k": bucket_of_k,
        "WIP": WIP,
        "D_long": D_long,
        "D_short": D_short,
        "D_eff": D_eff,
        "week1_days": week1_days,
        "Y": Y,
        "V": V,
        "R": R,
        "H": H,
        "A": A,
        "L_delay": L_delay,
        "gamma": gamma,
        "beta": beta,
    }