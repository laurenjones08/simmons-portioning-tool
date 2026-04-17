"""Unit tests for mix-detail helper behavior on the enumeration dashboard."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.enumeration_dashboard import _consume_mix_selection  # noqa: E402


def test_consume_mix_selection_opens_modal_for_new_selection():
    remembered_selection, mix_to_open, show_loading = _consume_mix_selection("mix-123", None)

    assert remembered_selection == "mix-123"
    assert mix_to_open == "mix-123"
    assert show_loading is True


def test_consume_mix_selection_ignores_same_selection():
    remembered_selection, mix_to_open, show_loading = _consume_mix_selection("mix-123", "mix-123")

    assert remembered_selection == "mix-123"
    assert mix_to_open is None
    assert show_loading is False


def test_consume_mix_selection_clears_when_selection_removed():
    remembered_selection, mix_to_open, show_loading = _consume_mix_selection(None, "mix-123")

    assert remembered_selection is None
    assert mix_to_open is None
    assert show_loading is False
