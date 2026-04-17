"""Centralized session-state helpers for the Streamlit UI.

Provides a small namespace for keys and helpers to read/write session state
in a consistent way.
"""
import streamlit as st
from typing import Any


def _ns(key: str) -> str:
    return f"ui::{key}"


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(_ns(key), default)


def set(key: str, value: Any) -> None:
    st.session_state[_ns(key)] = value


def clear(key: str) -> None:
    ns = _ns(key)
    if ns in st.session_state:
        del st.session_state[ns]
