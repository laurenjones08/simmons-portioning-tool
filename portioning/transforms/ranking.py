from __future__ import annotations

import pandas as pd


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    # Stable sorts: Upgrade desc then Trim asc
    if "Upgrade_%" in out.columns and "Trim_%" in out.columns:
        out = out.sort_values(["Upgrade_%", "Trim_%"], ascending=[False, True], kind="mergesort")
    elif "Upgrade_%" in out.columns:
        out = out.sort_values(["Upgrade_%"], ascending=[False], kind="mergesort")

    out = out.reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out
