from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import from config_manager instead of defining constants
from portioning.config_manager import load_config

# Load configuration at module level
_config = load_config()

# Expose configuration as module-level constants for backward compatibility
BUCKETS: List[Tuple[int, int]] = _config.buckets
ILLEGAL_PAIRS: Dict[str, List[str]] = _config.illegal_pairs

@dataclass(frozen=True)
class Defaults:
    trim_cap: int = 15

DEFAULTS = Defaults(
    trim_cap=_config.trim_cap
)


def reload_config():
    """Reload configuration from file (called after settings save)."""
    global _config, BUCKETS, ILLEGAL_PAIRS, DEFAULTS
    _config = load_config()
    BUCKETS = _config.buckets
    ILLEGAL_PAIRS = _config.illegal_pairs
    DEFAULTS = Defaults(
        trim_cap=_config.trim_cap
    )
