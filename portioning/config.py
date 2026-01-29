from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Buckets from your Simmons notebook (can be edited as needed).
BUCKETS: List[Tuple[int, int]] = [
    (0, 324),
    (325, 375),
    (376, 475),
    (476, 550),
    (551, 625),
    (626, 780),
    (390, 480),
    (481, 580),
]

# Illegal part pairing rules (Simmons notebook).
ILLEGAL_PAIRS: Dict[str, List[str]] = {
    "C": ["D"],
    "D": ["C", "T"],
    "R": ["V"],
    "V": ["R"],
    "M": ["K"],
    "K": ["M"],
    "T": ["D"],
}

@dataclass(frozen=True)
class Defaults:
    trim_cap: int = 15
    time_limit_sec: int = 60
    gap: float = 0.002
    chunk_size: int = 20

DEFAULTS = Defaults()
