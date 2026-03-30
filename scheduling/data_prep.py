def get_model_inputs():
    # --- Sets ---
    P = ["A", "B"]                 # SKUs
    T = [1, 2, 3]                  # Days
    K = ["cut", "pack"]            # Decisions
    L = ["line1", "line2"]         # Lines

    # --- WIP available each day ---
    WIP = {
        1: 1000,
        2: 1200,
        3: 1100,
    }

    # --- Demand (p, t) ---
    D = {
        ("A", 1): 400,
        ("A", 2): 500,
        ("A", 3): 450,
        ("B", 1): 300,
        ("B", 2): 350,
        ("B", 3): 400,
    }

    # --- Yield: how much SKU p comes from decision k ---
    Y = {
        ("A", "cut"): 0.6,
        ("A", "pack"): 0.4,
        ("B", "cut"): 0.5,
        ("B", "pack"): 0.5,
    }

    # --- Value per lb of decision ---
    V = {
        "cut": 2.0,
        "pack": 3.0,
    }

    # --- Rate (lbs/hour) ---
    R = {
        "cut": 100,
        "pack": 80,
    }

    # --- Hours available per line per day ---
    H = {
        ("line1", 1): 8,
        ("line1", 2): 8,
        ("line1", 3): 8,
        ("line2", 1): 6,
        ("line2", 2): 6,
        ("line2", 3): 6,
    }

    # --- Preferred line matrix A[k,l] ---
    A = {
        ("cut", "line1"): 1,
        ("cut", "line2"): 0,
        ("pack", "line1"): 0,
        ("pack", "line2"): 1,
    }

    # --- Lag (how many days demand can be delayed) ---
    L_delay = {
        "A": 1,
        "B": 1,
    }

    # --- Objective weights ---
    gamma = 0.1   # demand penalty
    beta = 5.0    # penalty for non-preferred line use

    return {
        "P": P,
        "T": T,
        "K": K,
        "L": L,
        "WIP": WIP,
        "D": D,
        "Y": Y,
        "V": V,
        "R": R,
        "H": H,
        "A": A,
        "L_delay": L_delay,
        "gamma": gamma,
        "beta": beta,
    }