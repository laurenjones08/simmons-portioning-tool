from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class EngineInput:
    df: pd.DataFrame

    # Common controls
    trim_cap: float

    # Enumeration controls
    bucket: Optional[tuple[int, int]] = None
    bird_size: str = "ALL"
    min_nuggets: int = 1
    customer_constraint: str = "NONE"  # NONE, RTL, FDS
    plant: Optional[str] = None
    pieces_per_min: float = 600.0
    line_eff: float = 0.85
    dsi_variance: float = 0.05  # Machine variance as decimal (5% default)


@dataclass(frozen=True)
class EngineResult:
    results_df: pd.DataFrame
    meta: Dict[str, Any]
    warnings: list[str]


class Engine:
    name: str = "base"

    def run(self, inp: EngineInput) -> EngineResult:  # pragma: no cover
        raise NotImplementedError
