from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import pandas as pd


class InputError(ValueError):
    pass


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


ENUM_REQUIRED = [
    "TradeNumber",
    "CustomerType",
    "ProductType",
    "TargetWeight",
    "BirdSize",
    "ProdPlant",
    "AllowedParts",
]



def validate_enumeration_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df)
    require_columns(df, ENUM_REQUIRED, "Enumeration engine")
    # Normalize key fields once.
    out = df.copy()
    out["CustomerType"] = out["CustomerType"].astype(str).str.upper().str.strip()
    out["BirdSize"] = out["BirdSize"].astype(str).str.upper().str.strip()
    out["ProdPlant"] = out["ProdPlant"].astype(str).str.upper().str.strip()
    out["TradeNumber"] = out["TradeNumber"].astype(str).str.strip()
    out["AllowedParts"] = out["AllowedParts"].fillna("").astype(str)
    out["TargetWeight"] = pd.to_numeric(out["TargetWeight"], errors="coerce").fillna(0).astype(float)
    return out
