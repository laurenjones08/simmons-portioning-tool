from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class LoadedInput:
    df: pd.DataFrame
    filename: str
    sheet_name: Optional[str]


def _rewind(uploaded_file) -> None:
    """Streamlit UploadedFile behaves like a file-like stream; rewind before re-reading."""
    try:
        uploaded_file.seek(0)
    except Exception:
        pass


def load_uploaded(uploaded_file, sheet_name: Optional[str] = None) -> LoadedInput:
    """Load a Streamlit uploaded file (CSV or Excel) into a DataFrame."""
    name = getattr(uploaded_file, "name", "uploaded")
    lower = name.lower()

    _rewind(uploaded_file)

    if lower.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return LoadedInput(df=df, filename=name, sheet_name=None)

    # Excel
    xls = pd.ExcelFile(uploaded_file)
    sheet = sheet_name or (xls.sheet_names[0] if xls.sheet_names else None)
    if sheet is None:
        raise ValueError("No sheets found in the uploaded Excel file.")

    _rewind(uploaded_file)
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    return LoadedInput(df=df, filename=name, sheet_name=sheet)


def list_excel_sheets(uploaded_file) -> Tuple[str, ...]:
    name = getattr(uploaded_file, "name", "uploaded")
    if not name.lower().endswith((".xlsx", ".xls")):
        return tuple()

    _rewind(uploaded_file)
    xls = pd.ExcelFile(uploaded_file)
    return tuple(xls.sheet_names)
