"""Multipage wrapper for the scheduling builder view."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme  # noqa: E402
from views.scheduling_create import render  # noqa: E402


st.set_page_config(page_title="Scheduling Create", layout="wide")
apply_theme()
render()

