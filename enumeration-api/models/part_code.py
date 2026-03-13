from enum import Enum
from typing import Dict


# Module-level mapping for human-readable descriptions of part codes.
_PART_DESCRIPTIONS: Dict[str, str] = {
    "D": "DSI Tail/DSI Nugget",
    "R": "DSI/A3D3 Membrane Piece",
    "M": "DSI/A3D3 Non-Membrane Piece",
    "T": "DB20 Tail",
    "V": "DB20 Membrane Piece",
    "K": "DB20 Non-Membrane Piece",
    "S": "Slit Top",
    "U": "Slit Bottom",
    "C": "Customer Specific DSI Portion (slit, template, etc)",
    "J": "DSI Portioned Slit Top",
    "W": "DSI Portioned Slit Bottom",
    "G": "I-cut",
}


class PartCode(str, Enum):
    """Enumeration of part codes used in cut strategies and mixes.

    The `value` of each enum is the single-letter part code (used in data and references).
    Use the `.description` property for a human-readable label (UI/visualization).
    """

    D = "D"
    R = "R"
    M = "M"
    T = "T"
    V = "V"
    K = "K"
    S = "S"
    U = "U"
    C = "C"
    J = "J"
    W = "W"
    G = "G"

    @property
    def description(self) -> str:
        """Human-friendly description for UI/visualization."""
        return _PART_DESCRIPTIONS.get(self.value, "")

    def __str__(self) -> str:
        return self.value
