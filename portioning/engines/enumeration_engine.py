from __future__ import annotations

from typing import List, Tuple
import itertools as it

import pandas as pd

from portioning.engines.base import Engine, EngineInput, EngineResult
from portioning.schemas import validate_enumeration_df
from portioning.config import ILLEGAL_PAIRS


class EnumerationEngine(Engine):
    """
    Enumeration-based portioning engine (interactive / exploratory).

    This engine enumerates combinations of SKUs (group sizes 1..4) and filters them
    according to:

    1) Input constraints (UI controls)
       - Bucket: total target weight must fall within [bmin, bmax]
       - Bird size: either ALL, or match specific bird size (allowing SKU BirdSize=ALL)
       - Plant: if chosen, all SKUs must come from that plant
       - Customer constraint: NONE, RTL, or FDS (must include at least one of that type)
       - Trim cap: computed trim% must be <= trim_cap
       - Upgrade cap: upgrade% must be <= 100

    2) Cut feasibility constraints (from the Simmons notebook)
       - Each SKU has AllowedParts: a comma-separated list of acceptable part codes
       - Each SKU must be assigned a *unique* part code in the group
       - Illegal part pairings are disallowed (ILLEGAL_PAIRS)

    Metrics:
    - TargetSum_g: sum(TargetWeight) for the group
    - Upgrade_%: (TargetSum_g / bucket_target) * 100
    - Trim_%: 100 - Upgrade_%

    New columns (requested):
    - Filet_Count: number of SKUs in the group whose ProductType is filet-ish
    - Nugget_Count: number of SKUs in the group whose ProductType is nugget-ish
    - PartCounts: display string e.g. "FILETx2, NUGGETx1"

    Performance notes:
    - Avoids repeated pandas slicing inside the combinations loop.
    - Pre-extracts arrays/lists (TradeNumber, CustomerType, ProductType, TargetWeight, AllowedParts).
    - Uses early pruning before running cut assignment (most expensive step).
    - Includes a MAX_CANDIDATES guardrail to avoid combinatorial blow-ups.
    """

    name = "enumeration"

    # Treat these ProductType values as "filet"
    FILET_ALIASES = {"FILET", "FILLET", "BREAST FILET", "BREAST FILLET"}
    # Treat these ProductType values as "nugget"
    NUGGET_ALIASES = {"NUGGET", "NUGGETS"}

    def run(self, inp: EngineInput) -> EngineResult:
        """
        Run the enumeration engine.

        Parameters
        ----------
        inp:
            EngineInput containing the uploaded dataframe and UI controls.
            Requires inp.bucket to be set.

        Returns
        -------
        EngineResult
            - results_df: DataFrame of valid combinations
            - meta: run metadata useful for debugging and UI display
            - warnings: user-facing warnings (e.g., too many candidates, no solutions)
        """
        df = validate_enumeration_df(inp.df)

        if inp.bucket is None:
            raise ValueError("Enumeration engine requires 'bucket'.")

        # ---- Normalize controls ----
        plant = (inp.plant or "").upper().strip() or None
        (bmin, bmax) = inp.bucket
        # Bucket "target" is the midpoint used for upgrade/trim calculation (matches your notebook)
        btarget = ((bmax - bmin) / 2.0) + bmin

        bird_size = (inp.bird_size or "ALL").upper().strip()
        customer_rule = (inp.customer_constraint or "NONE").upper().strip()
        max_trim = float(inp.trim_cap)
        min_nuggets = int(inp.min_nuggets)

        # -----------------------------
        # PRE-FILTER (MAJOR SPEEDUP)
        # -----------------------------
        # Reduce the candidate SKU list BEFORE enumerating combinations.
        # This is the single most important lever to keep compute reasonable.
        mask = pd.Series(True, index=df.index)

        if plant is not None:
            mask &= (df["ProdPlant"] == plant)

        if bird_size != "ALL":
            # Allow SKU BirdSize=ALL as compatible with SB/BB selections (matches original logic)
            mask &= df["BirdSize"].isin([bird_size, "ALL"])

        # Note: We do NOT filter to only RTL or only FDS here, because groups can mix.
        # We enforce "must include at least one" later in passes_customer_rule().
        if customer_rule in {"RTL", "FDS"}:
            # Keep only plausible types if your data contains other customer labels.
            mask &= df["CustomerType"].isin(["RTL", "FDS", "ALL"])

        df = df.loc[mask].reset_index(drop=True)

        # Guardrail: enumeration grows rapidly with candidate size (n choose k).
        # If this triggers, you should tighten filters (plant/bird/customer/bucket),
        # or switch to the two-stage engine for production-scale runs.
        MAX_CANDIDATES = 80
        if len(df) > MAX_CANDIDATES:
            return EngineResult(
                results_df=pd.DataFrame(),
                meta={
                    "bucket": inp.bucket,
                    "bucket_target": btarget,
                    "bird_size": bird_size,
                    "trim_cap": max_trim,
                    "customer_constraint": customer_rule,
                    "plant": plant or "ALL",
                    "candidate_rows": int(len(df)),
                    "max_candidates": MAX_CANDIDATES,
                },
                warnings=[
                    f"Too many candidate rows after filters ({len(df)}). "
                    f"Add more filters (plant/bird/customer) or lower MAX_CANDIDATES."
                ],
            )

        # -----------------------------
        # PRE-EXTRACT ARRAYS (NO PANDAS IN HOT LOOP)
        # -----------------------------
        trade = df["TradeNumber"].astype(str).to_list()
        cust_type = df["CustomerType"].astype(str).str.upper().str.strip().to_list()

        # Normalize ProductType once for fast part counting
        product_type = (
            df["ProductType"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .to_list()
        )

        is_nugget = [pt in self.NUGGET_ALIASES for pt in product_type]
        min_nuggets = int(getattr(inp, "min_nuggets", 0) or 0)

        weights = df["TargetWeight"].astype(float).to_numpy()

        # Parse AllowedParts once (list of valid part codes per SKU)
        allowed_parts: List[List[str]] = [
            [c.strip() for c in str(s).split(",") if c.strip()]
            for s in df["AllowedParts"].fillna("").astype(str).to_list()
        ]

        # Precompute illegal lookup: dict[str, set[str]] for quick intersection checks
        illegal_map = {k: set(v) for k, v in ILLEGAL_PAIRS.items()}

        # -----------------------------
        # Helpers (hot-path safe)
        # -----------------------------
        def passes_customer_rule(idxs: Tuple[int, ...]) -> bool:
            """
            Enforce 'must include' customer constraints:
            - NONE: always pass
            - RTL: at least one SKU has CustomerType == RTL
            - FDS: at least one SKU has CustomerType == FDS
            """
            if customer_rule == "NONE":
                return True
            if customer_rule == "RTL":
                return any(cust_type[i] == "RTL" for i in idxs)
            if customer_rule == "FDS":
                return any(cust_type[i] == "FDS" for i in idxs)
            return True

        def count_parts(idxs: Tuple[int, ...]) -> tuple[int, int]:
            """
            Compute (filet_count, nugget_count) for the combination.

            Mapping:
            - ProductType in FILET_ALIASES => filet
            - ProductType in NUGGET_ALIASES => nugget

            Unknown product types are ignored in both counts.
            """
            filet = 0
            nugget = 0
            for i in idxs:
                pt = product_type[i]
                if pt in self.FILET_ALIASES:
                    filet += 1
                elif pt in self.NUGGET_ALIASES:
                    nugget += 1
            return filet, nugget

        # -----------------------------
        # MAIN ENUMERATION (FAST PATH)
        # -----------------------------
        rows: List[dict] = []
        n = len(df)

        for k in range(1, 5):
            for idxs in it.combinations(range(n), k):
                # 1) Must-include customer type constraint
                if not passes_customer_rule(idxs):
                    continue

                # 2) Weight sum and bucket range
                # Use numpy array sum for speed
                # Compute base sum (1 unit each)
                base_sum = float(weights[list(idxs)].sum())

                # Build unit plan (default 1 each)
                unit_map = {trade[i]: 1 for i in idxs}

                # If the combo contains nuggets, enforce min_nuggets per nugget SKU
                nugget_idxs = [i for i in idxs if is_nugget[i]]

                if nugget_idxs and min_nuggets > 0:
                    # Apply minimum requirement: each nugget SKU must be at least min_nuggets units
                    min_required_sum = base_sum
                    for i in nugget_idxs:
                        extra_units_needed = min_nuggets - 1  # because we already counted 1 unit in base_sum
                        if extra_units_needed > 0:
                            min_required_sum += extra_units_needed * float(weights[i])
                            unit_map[trade[i]] = min_nuggets

                    # If even the minimum required production doesn't fit, reject
                    if min_required_sum > bmax:
                        continue

                    # If minimum is below bucket min, we can optionally add more nugget units to reach bucket
                    # or to move closer to bucket target without exceeding bmax.
                    t_sum = min_required_sum

                    # Add extra nugget units greedily until we reach bmin (or just improve closeness to btarget)
                    # This is optional but matches "can produce more than min_nuggets".
                    if t_sum < bmin:
                        # We need at least (bmin - t_sum) more grams.
                        # Greedy: add units of the heaviest nugget SKU first (fewer units needed).
                        nugget_idxs_sorted = sorted(nugget_idxs, key=lambda i: float(weights[i]), reverse=True)
                        for i in nugget_idxs_sorted:
                            wi = float(weights[i])
                            if wi <= 0:
                                continue
                            # Max additional units we can add for this SKU without exceeding bmax
                            max_add = int((bmax - t_sum) // wi)
                            if max_add <= 0:
                                continue
                            # Add only what we need to reach bmin (rounded up)
                            needed = bmin - t_sum
                            add = min(max_add, int((needed + wi - 1) // wi))  # ceil(needed/wi)
                            if add > 0:
                                t_sum += add * wi
                                unit_map[trade[i]] += add
                            if t_sum >= bmin:
                                break

                    # After optional fill, still must fit in bucket
                    if not (bmin <= t_sum <= bmax):
                        continue

                else:
                    # No nuggets (or min_nuggets==0): normal behavior
                    t_sum = base_sum
                    if not (bmin <= t_sum <= bmax):
                        continue

                # 3) Upgrade/Trim metrics
                upgrade = (t_sum / btarget) * 100.0
                if upgrade > 100.0:
                    continue

                trim = 100.0 - upgrade
                if trim > max_trim:
                    continue

                # 4) CUT ASSIGNMENT (most expensive step)
                # Assign each SKU a unique cut from its AllowedParts list.
                used = set()
                cut_data = []
                ok = True

                for i in idxs:
                    cuts = allowed_parts[i]
                    if not cuts:
                        ok = False
                        break

                    chosen = None
                    for c in cuts:
                        if c not in used:
                            chosen = c
                            used.add(c)
                            break

                    if chosen is None:
                        ok = False
                        break

                    cut_data.append((trade[i], chosen))

                if not ok:
                    continue

                # 6) Illegal pairing check (fast set intersection)
                for c in used:
                    bad = illegal_map.get(c)
                    if bad and (used & bad):
                        ok = False
                        break

                if not ok:
                    continue

                # 7) Emit row
                rows.append(
                    {
                        "GroupSize": k,
                        "SKU_IDs": [trade[i] for i in idxs],
                        "CustomerTypes": [cust_type[i] for i in idxs],
                        "UnitsBySKU": unit_map,
                        "UnitPlan": ", ".join(f"{sku}x{u}" for sku, u in unit_map.items()),
                        "Cut_Data": cut_data,
                        "TargetSum_g": t_sum,
                        "Upgrade_%": upgrade,
                        "Trim_%": trim,
                    }
                )

        out_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        meta = {
            "bucket": inp.bucket,
            "bucket_target": btarget,
            "bird_size": bird_size,
            "trim_cap": max_trim,
            "min_nuggets": min_nuggets,
            "customer_constraint": customer_rule,
            "plant": plant or "ALL",
            "candidate_rows": int(len(df)),
            "valid_combinations": int(len(out_df)) if not out_df.empty else 0,
        }

        warnings = []
        if out_df.empty:
            warnings.append("No valid combinations found. Tighten/loosen filters and try again.")

        return EngineResult(results_df=out_df, meta=meta, warnings=warnings)
