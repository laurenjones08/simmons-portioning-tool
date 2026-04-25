"""Regression tests for shared Streamlit theme CSS."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui import theme  # noqa: E402


def test_apply_theme_resets_dialog_horizontal_blocks(monkeypatch):
    captured = {}

    def fake_markdown(body, unsafe_allow_html=False):
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(theme.st, "markdown", fake_markdown)

    theme.apply_theme()

    css = captured["body"]
    assert captured["unsafe_allow_html"] is True
    assert '[data-testid="stDialog"] [data-testid="stHorizontalBlock"]' in css
    assert "position: static !important;" in css
    assert "background: transparent !important;" in css
    assert "height: auto !important;" in css
    assert "max-height: calc(100vh - 24px) !important;" in css
    assert "margin: 0 0 14px 0;" in css
    assert "margin-bottom: 14px !important;" in css
    assert "--navbar-bg:" in css
    assert "--theme-icon-bg:" in css
    assert ".simmons-kpi-flex-row" in css
    assert '[data-testid="stDialog"] [data-testid="stDataFrame"]' in css
    assert ".simmons-detail-config-card" in css
    assert ".simmons-dialog-footer-divider" in css
    assert '[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"]' in css
    assert ".sfy-navbar-shell" in css
    assert ".sfy-page-header-shell" in css
    assert "[data-testid=\"element-container\"]:has(.sfy-page-header-shell)" in css
    assert "text-align: left;" in css


def test_apply_theme_uses_dark_mode_variables(monkeypatch):
    captured = {}

    def fake_markdown(body, unsafe_allow_html=False):
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(theme.st, "markdown", fake_markdown)
    monkeypatch.setattr(theme.st, "session_state", {"ui_theme_mode": "dark"})

    theme.apply_theme()

    css = captured["body"]
    assert captured["unsafe_allow_html"] is True
    assert "--page-bg:         #0B1220;" in css
    assert "--card-bg:         #142033;" in css
    assert "--dialog-bg:       #122033;" in css
    assert "--navbar-text:     #DCE7F8;" in css
