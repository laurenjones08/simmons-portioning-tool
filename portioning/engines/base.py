from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class EngineInput:
    df: pd.DataFrame
    engine_name: str

    # Common controls
    trim_cap: float

    # Enumeration controls
    bucket: Optional[tuple[int, int]] = None
    bird_size: str = "ALL"
    min_nuggets: int = 1
    customer_constraint: str = "NONE"  # NONE, RTL, FDS
    plant: Optional[str] = None

    # Two-stage controls
    sheet_name: str = "Sheet1"
    use_month_config: bool = False
    time_limit_sec: int = 60
    gap: float = 0.002
    chunk_size: int = 20
    pieces_per_min: float = 600.0
    line_eff: float = 0.85


@dataclass(frozen=True)
class EngineResult:
    results_df: pd.DataFrame
    meta: Dict[str, Any]
    warnings: list[str]


class Engine:
    name: str = "base"

    def run(self, inp: EngineInput) -> EngineResult:  # pragma: no cover
        raise NotImplementedError
