from __future__ import annotations

import pandas as pd


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize engine or raw input output into UI-friendly columns using
    the actual Simmons CSV column names.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["Rank", "SKU_IDs", "CustomerTypes", "TargetSum_g", "Upgrade_%", "Trim_%"]
        )

    out = df.copy()

    # ----------------------------
    # SKU IDs
    # ----------------------------
    if "SKU_IDs" not in out.columns:
        if "TradeNumber" in out.columns:
            out["SKU_IDs"] = out["TradeNumber"].astype(str)
        else:
            out["SKU_IDs"] = out.index.astype(str)

    out["SKU_IDs"] = out["SKU_IDs"].apply(
        lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    )

    # ----------------------------
    # Customer types
    # ----------------------------
    if "CustomerTypes" not in out.columns:
        if "CustomerType" in out.columns:
            out["CustomerTypes"] = out["CustomerType"].astype(str)
        else:
            out["CustomerTypes"] = ""

    out["CustomerTypes"] = out["CustomerTypes"].apply(
        lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    )

    # ----------------------------
    # Target weight (grams)
    # ----------------------------
    if "TargetSum_g" not in out.columns:
        if "TargetWeight" in out.columns:
            out["TargetSum_g"] = pd.to_numeric(out["TargetWeight"], errors="coerce")
        else:
            out["TargetSum_g"] = pd.NA

    # ----------------------------
    # Upgrade / Trim
    # ----------------------------
    # Enumeration engine already supplies these
    if "Upgrade_%" not in out.columns:
        if "Upgrade" in out.columns:
            out["Upgrade_%"] = pd.to_numeric(out["Upgrade"], errors="coerce")
        else:
            out["Upgrade_%"] = pd.NA

    if "Trim_%" not in out.columns:
        if "Trim" in out.columns:
            out["Trim_%"] = pd.to_numeric(out["Trim"], errors="coerce")
        else:
            out["Trim_%"] = pd.NA

    # Ensure numeric
    for c in ["TargetSum_g", "Upgrade_%", "Trim_%"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # ----------------------------
    # Final column order
    # ----------------------------
    core = ["SKU_IDs", "CustomerTypes", "TargetSum_g", "Upgrade_%", "Trim_%"]
    extras = [c for c in out.columns if c not in core]
    out = out[core + extras]

    return out
