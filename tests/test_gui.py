"""
Tests for the Tkinter GUI.

The window is never shown: the Tk root is created and immediately withdrawn,
so these tests exercise input parsing and result binding headlessly. Modal
messageboxes are monkeypatched so nothing blocks on user interaction.
"""

from __future__ import annotations

import contextlib
import importlib.util

import pytest

from black_scholes import OptionParams, OptionType

if importlib.util.find_spec("tkinter") is None:  # pragma: no cover - build dependent
    pytest.skip("tkinter is not available", allow_module_level=True)

import tkinter as tk

from black_scholes import gui

# The GUI uses an em dash as its "no value" placeholder.
BLANK = "\u2014"

GREEK_KEYS = ("delta", "gamma", "vega", "theta", "rho")

# --- Test Fixtures ---


@pytest.fixture
def root():
    """A hidden Tk root; skipped when no display is available."""
    try:
        widget = tk.Tk()
    except Exception as exc:  # pragma: no cover - headless environments
        pytest.skip(f"no usable Tk display: {exc}")

    widget.withdraw()
    try:
        yield widget
    finally:
        with contextlib.suppress(Exception):
            widget.destroy()


@pytest.fixture
def app(root, monkeypatch):
    """A BlackScholesGUI whose dialogs are stubbed out."""
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: shown.append(("error", a)))
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: shown.append(("info", a)))

    widget = gui.BlackScholesGUI(root)
    root.withdraw()
    widget.shown = shown
    return widget


def _set_field(app, key: str, value: str) -> None:
    entry = app.entries[key]
    entry.delete(0, tk.END)
    entry.insert(0, value)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# --- Parameter Parsing ---


class TestGetParams:
    """Input parsing and validation."""

    def test_defaults_produce_valid_params(self, app):
        params = app._get_params()
        assert isinstance(params, OptionParams)
        assert params.spot == 100.0
        assert params.strike == 100.0
        assert params.time_to_maturity == 1.0
        assert params.risk_free_rate == 0.05
        assert params.volatility == 0.20
        assert params.dividend_yield == 0.0
        assert params.option_type is OptionType.CALL

    @pytest.mark.parametrize("opt_type", ["call", "put"])
    def test_option_type_radio(self, app, opt_type):
        app.option_type_var.set(opt_type)
        params = app._get_params()
        assert params.option_type is OptionType(opt_type)

    def test_all_numeric_fields_are_wired(self, app):
        assert set(app.entries.keys()) == {"spot", "strike", "time", "rate", "vol", "div_yield"}

    def test_negative_spot_returns_none(self, app):
        _set_field(app, "spot", "-100")
        assert app._get_params() is None

    def test_zero_strike_returns_none(self, app):
        _set_field(app, "strike", "0")
        assert app._get_params() is None

    def test_zero_time_returns_none(self, app):
        _set_field(app, "time", "0")
        assert app._get_params() is None

    def test_zero_vol_returns_none(self, app):
        _set_field(app, "vol", "0")
        assert app._get_params() is None

    def test_negative_rate_returns_none(self, app):
        _set_field(app, "rate", "-0.01")
        assert app._get_params() is None

    def test_negative_dividend_returns_none(self, app):
        _set_field(app, "div_yield", "-0.01")
        assert app._get_params() is None

    def test_non_numeric_text_returns_none(self, app):
        _set_field(app, "vol", "twenty percent")
        assert app._get_params() is None

    def test_blank_field_returns_none(self, app):
        _set_field(app, "spot", "")
        assert app._get_params() is None

    def test_invalid_input_shows_error_dialog(self, app):
        _set_field(app, "spot", "-1")
        app._get_params()
        assert app.shown
        assert app.shown[-1][0] == "error"
        assert "Invalid Input" in app.shown[-1][1][0]

    def test_valid_input_shows_no_dialog(self, app):
        app._get_params()
        assert app.shown == []

    def test_custom_values_are_parsed(self, app):
        _set_field(app, "spot", "120")
        _set_field(app, "strike", "110")
        _set_field(app, "time", "0.5")
        _set_field(app, "rate", "0.03")
        _set_field(app, "vol", "0.35")
        _set_field(app, "div_yield", "0.01")

        params = app._get_params()
        assert params.spot == 120.0
        assert params.strike == 110.0
        assert params.time_to_maturity == 0.5
        assert params.risk_free_rate == 0.03
        assert params.volatility == 0.35
        assert params.dividend_yield == 0.01


# --- Price Calculation ---


class TestOnPrice:
    """Binding calculated values into the display variables."""

    def test_price_var_populated(self, app):
        assert app.price_var.get() == BLANK
        app._on_price()
        assert _is_number(app.price_var.get())
        assert float(app.price_var.get()) == pytest.approx(10.450584, abs=1e-6)

    def test_all_greeks_populated(self, app):
        app._on_price()
        assert set(app.greek_vars.keys()) == set(GREEK_KEYS)
        for key in GREEK_KEYS:
            value = app.greek_vars[key].get()
            assert value != BLANK
            assert _is_number(value), f"{key} is not numeric: {value!r}"

    def test_greek_signs_match_convention(self, app):
        app._on_price()
        assert float(app.greek_vars["delta"].get()) > 0
        assert float(app.greek_vars["gamma"].get()) > 0
        assert float(app.greek_vars["vega"].get()) > 0
        assert float(app.greek_vars["rho"].get()) > 0

    def test_put_delta_is_negative(self, app):
        app.option_type_var.set("put")
        app._on_price()
        assert float(app.price_var.get()) == pytest.approx(5.573526, abs=1e-6)
        assert float(app.greek_vars["delta"].get()) < 0

    def test_status_becomes_done(self, app):
        app._on_price()
        assert app.status_var.get() == "Done"

    def test_monte_carlo_label_empty_when_disabled(self, app):
        app.mc_var.set(False)
        app._on_price()
        assert app.mc_var_label.get() == ""

    def test_invalid_input_leaves_display_untouched(self, app):
        _set_field(app, "spot", "-100")
        app._on_price()
        displayed = {key: app.greek_vars[key].get() for key in GREEK_KEYS}
        displayed["price"] = app.price_var.get()
        assert displayed == {"price": BLANK, **dict.fromkeys(GREEK_KEYS, BLANK)}

    def test_invalid_input_reports_error(self, app):
        _set_field(app, "vol", "0")
        app._on_price()
        assert app.shown
        assert app.shown[-1][0] == "error"

    def test_modified_inputs_are_priced(self, app):
        _set_field(app, "spot", "120")
        _set_field(app, "strike", "120")
        _set_field(app, "time", "2")
        _set_field(app, "rate", "0.04")
        _set_field(app, "vol", "0.25")
        app._on_price()
        assert _is_number(app.price_var.get())
        assert float(app.price_var.get()) > 0


# --- Clear ---


class TestOnClear:
    """Resetting the form."""

    def test_price_var_reset(self, app):
        app._on_price()
        assert app.price_var.get() != BLANK
        app._on_clear()
        assert app.price_var.get() == BLANK

    def test_greeks_reset(self, app):
        app._on_price()
        app._on_clear()
        for key in GREEK_KEYS:
            assert app.greek_vars[key].get() == BLANK

    def test_fields_restored_to_defaults(self, app):
        _set_field(app, "spot", "500")
        _set_field(app, "vol", "1.5")
        app.option_type_var.set("put")
        app._on_clear()

        assert app.entries["spot"].get() == "100.0"
        assert app.entries["strike"].get() == "100.0"
        assert app.entries["time"].get() == "1.0"
        assert app.entries["rate"].get() == "0.05"
        assert app.entries["vol"].get() == "0.20"
        assert app.entries["div_yield"].get() == "0.0"
        assert app.option_type_var.get() == "call"

    def test_monte_carlo_flag_and_label_reset(self, app):
        app.mc_var.set(True)
        app._on_clear()
        assert app.mc_var.get() is False
        assert app.mc_var_label.get() == ""

    def test_status_shows_cleared(self, app):
        app._on_clear()
        assert app.status_var.get() == "Cleared"

    def test_clear_then_price_still_works(self, app):
        app._on_clear()
        app._on_price()
        assert _is_number(app.price_var.get())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
