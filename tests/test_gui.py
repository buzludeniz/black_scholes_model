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


def _is_destroyed(window) -> bool:
    """True once ``window`` has been torn down.

    Tk answers ``winfo_exists`` with 0 for a destroyed widget, and with a
    ``TclError`` on some builds, so both spellings count as destroyed.
    """
    try:
        return not window.winfo_exists()
    except tk.TclError:
        return True


def _stub_market_price_dialog(monkeypatch, app, result):
    """Replace the modal price prompt with a stub that never blocks.

    ``BlackScholesGUI._on_iv`` calls ``root.wait_window(dialog)``, which returns
    only once the dialog is destroyed. Letting that run for real would hang the
    suite, so the dialog class is swapped for one whose ``result`` is fixed and
    ``wait_window`` is neutralised. Returns the list of opened stubs.
    """
    opened = []

    class StubDialog:
        def __init__(self, parent, title, prompt):
            self.parent = parent
            self.title = title
            self.prompt = prompt
            self.result = result
            opened.append(self)

    monkeypatch.setattr(gui, "MarketPriceDialog", StubDialog)
    monkeypatch.setattr(app.root, "wait_window", lambda *a, **k: None)
    return opened


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


# --- Monte Carlo Validation ---


class TestOnPriceMonteCarlo:
    """The optional Monte Carlo cross-check inside ``_on_price``."""

    def test_monte_carlo_label_populated_when_enabled(self, app):
        app.mc_var.set(True)
        app._on_price()

        label = app.mc_var_label.get()
        assert label != ""
        assert "MC=" in label
        assert "OK" in label or "OUT OF TOLERANCE" in label
        assert app.status_var.get() == "Done"

    def test_monte_carlo_price_tracks_the_analytic_price(self, app):
        app.mc_var.set(True)
        app._on_price()

        mc_price = float(app.mc_var_label.get().split("MC=")[1].split()[0])
        assert mc_price == pytest.approx(10.450584, abs=0.5)

    def test_seeded_sampling_is_deterministic(self, app):
        app.mc_var.set(True)
        app._on_price()
        first = app.mc_var_label.get()
        app._on_price()
        assert app.mc_var_label.get() == first

    def test_close_agreement_is_reported_as_ok(self, app, monkeypatch):
        monkeypatch.setattr(gui, "monte_carlo_price", lambda *a, **k: (10.450584, 0.01))
        app.mc_var.set(True)
        app._on_price()
        assert app.mc_var_label.get().endswith("OK")

    def test_wild_disagreement_is_flagged(self, app, monkeypatch):
        monkeypatch.setattr(gui, "monte_carlo_price", lambda *a, **k: (50.0, 0.01))
        app.mc_var.set(True)
        app._on_price()
        assert app.mc_var_label.get().endswith("OUT OF TOLERANCE")

    def test_tolerance_scales_with_the_standard_error(self, app, monkeypatch):
        # Same 10.0 gap, but a wide standard error keeps it inside tolerance.
        monkeypatch.setattr(gui, "monte_carlo_price", lambda *a, **k: (0.450584, 10.0))
        app.mc_var.set(True)
        app._on_price()
        assert app.mc_var_label.get().endswith("OK")

    def test_runs_a_hundred_thousand_seeded_paths(self, app, monkeypatch):
        seen = {}

        def fake_mc(params, *, n_paths, seed):
            seen["n_paths"] = n_paths
            seen["seed"] = seed
            return 10.450584, 0.05

        monkeypatch.setattr(gui, "monte_carlo_price", fake_mc)
        app.mc_var.set(True)
        app._on_price()
        assert seen == {"n_paths": 100_000, "seed": 0}

    def test_price_and_greeks_still_populate(self, app):
        app.mc_var.set(True)
        app._on_price()
        assert float(app.price_var.get()) == pytest.approx(10.450584, abs=1e-6)
        for key in GREEK_KEYS:
            assert _is_number(app.greek_vars[key].get())

    def test_label_is_cleared_when_the_box_is_unticked(self, app):
        app.mc_var.set(True)
        app._on_price()
        assert app.mc_var_label.get() != ""

        app.mc_var.set(False)
        app._on_price()
        assert app.mc_var_label.get() == ""


# --- Implied Volatility ---


class TestOnIv:
    """Inverting an observed market price back into a volatility."""

    def test_solved_volatility_is_written_to_the_vol_field(self, app, monkeypatch):
        # 10.450584 is the analytic price of the default inputs at vol 0.20,
        # so the inversion must return the value it started from.
        _stub_market_price_dialog(monkeypatch, app, 10.450584)
        app._on_iv()
        assert app.entries["vol"].get() == "0.200000"

    def test_status_reports_the_solved_volatility(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, 10.450584)
        app._on_iv()

        status = app.status_var.get()
        assert status.startswith("Implied vol:")
        assert status == "Implied vol: 20.00%"

    def test_success_shows_an_info_dialog(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, 10.450584)
        app._on_iv()

        assert app.shown
        assert app.shown[-1][0] == "info"
        assert app.shown[-1][1][0] == "Implied Volatility"

    def test_dialog_asks_for_a_market_price(self, app, monkeypatch):
        opened = _stub_market_price_dialog(monkeypatch, app, 10.450584)
        app._on_iv()

        assert len(opened) == 1
        assert opened[0].parent is app.root
        assert opened[0].title == "Implied Volatility"
        assert opened[0].prompt == "Enter observed market price:"

    def test_high_market_price_implies_a_higher_volatility(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, 25.0)
        app._on_iv()

        vol = float(app.entries["vol"].get())
        assert vol > 0.20
        assert "Implied vol:" in app.status_var.get()

    def test_cancel_leaves_inputs_untouched(self, app, monkeypatch):
        _set_field(app, "vol", "0.42")
        _stub_market_price_dialog(monkeypatch, app, None)
        app._on_iv()

        assert app.status_var.get() == "Implied volatility cancelled"
        assert app.entries["vol"].get() == "0.42"
        assert app.shown == []

    def test_cancel_shows_no_dialog(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, None)
        app._on_iv()
        assert app.shown == []

    def test_arbitrage_violating_price_reports_an_error(self, app, monkeypatch):
        # 500.0 sits above the call's no-arbitrage ceiling of 100.0.
        _stub_market_price_dialog(monkeypatch, app, 500.0)
        app._on_iv()

        assert app.status_var.get() == "Implied volatility failed"
        assert app.shown[-1][0] == "error"
        assert app.shown[-1][1][0] == "Implied Volatility Error"

    def test_arbitrage_failure_leaves_the_vol_field_untouched(self, app, monkeypatch):
        _set_field(app, "vol", "0.42")
        _stub_market_price_dialog(monkeypatch, app, 500.0)
        app._on_iv()

        assert app.entries["vol"].get() == "0.42"
        assert app.shown[-1][0] == "error"

    def test_arbitrage_failure_shows_no_info_dialog(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, 500.0)
        app._on_iv()
        assert all(kind != "info" for kind, _ in app.shown)

    def test_solved_volatility_round_trips_through_the_form(self, app, monkeypatch):
        _stub_market_price_dialog(monkeypatch, app, 25.0)
        app._on_iv()
        _set_field(app, "vol", app.entries["vol"].get())
        app._on_price()

        assert float(app.price_var.get()) == pytest.approx(25.0, abs=1e-3)

    def test_invalid_params_never_open_the_dialog(self, app, monkeypatch):
        opened = _stub_market_price_dialog(monkeypatch, app, 10.450584)
        _set_field(app, "vol", "0")
        app._on_iv()

        assert opened == []
        assert app.status_var.get() == "Ready"
        assert app.shown[-1][0] == "error"

    def test_put_params_are_inverted(self, app, monkeypatch):
        app.option_type_var.set("put")
        _stub_market_price_dialog(monkeypatch, app, 5.573526)
        app._on_iv()

        assert app.entries["vol"].get() == "0.200000"
        assert app.status_var.get().startswith("Implied vol:")


# --- Market Price Dialog ---


@pytest.fixture
def dialog(root, monkeypatch):
    """A real MarketPriceDialog on a hidden parent, destroyed on teardown."""
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: None)

    window = gui.MarketPriceDialog(root, "Implied Volatility", "Enter observed market price:")
    try:
        yield window
    finally:
        with contextlib.suppress(Exception):
            window.destroy()


class TestMarketPriceDialog:
    """The modal that collects an observed market price."""

    def test_starts_with_no_result_and_an_empty_field(self, dialog):
        assert dialog.result is None
        assert dialog.entry.get() == ""

    def test_window_is_configured(self, dialog):
        assert dialog.title() == "Implied Volatility"
        assert dialog.wm_resizable() == (0, 0)
        assert str(dialog.transient()) == str(dialog.master)

    def test_ok_accepts_a_decimal_price(self, dialog):
        dialog.entry.insert(0, "12.5")
        dialog._on_ok()

        assert dialog.result == pytest.approx(12.5)
        assert _is_destroyed(dialog)

    def test_ok_accepts_an_integer_price(self, dialog):
        dialog.entry.insert(0, "42")
        dialog._on_ok()

        assert dialog.result == pytest.approx(42.0)
        assert _is_destroyed(dialog)

    def test_ok_rejects_non_numeric_input(self, dialog, monkeypatch):
        errors = []
        monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))

        dialog.entry.insert(0, "not a number")
        dialog._on_ok()

        assert dialog.result is None
        assert not _is_destroyed(dialog)
        assert errors
        assert errors[-1][0] == "Invalid Input"
        assert errors[-1][1] == "Please enter a valid number"

    def test_ok_rejects_empty_input(self, dialog, monkeypatch):
        errors = []
        monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))

        dialog._on_ok()

        assert dialog.result is None
        assert not _is_destroyed(dialog)
        assert errors

    def test_rejected_input_can_be_corrected_and_accepted(self, dialog, monkeypatch):
        errors = []
        monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))

        dialog.entry.insert(0, "oops")
        dialog._on_ok()
        dialog.entry.delete(0, tk.END)
        dialog.entry.insert(0, "3.5")
        dialog._on_ok()

        assert dialog.result == pytest.approx(3.5)
        assert errors

    def test_cancel_discards_the_entry_contents(self, dialog):
        dialog.entry.insert(0, "12.5")
        dialog._on_cancel()

        assert dialog.result is None
        assert _is_destroyed(dialog)

    def test_return_and_escape_are_bound(self, dialog):
        # Key delivery is not exercised: Tk only dispatches generated events to
        # bindings on a mapped window, and this dialog's parent stays hidden.
        # Asserting the bindings are registered keeps the line covered without
        # depending on window visibility.
        assert dialog.bind("<Return>")
        assert dialog.bind("<Escape>")


# --- Entry Point ---


class TestRunGui:
    """``run_gui`` wires a root, the app, and the mainloop together."""

    @staticmethod
    def _patch_gui(monkeypatch):
        """Replace Tk and the app class with recorders; return the log."""
        log = []

        class FakeRoot:
            def __init__(self):
                log.append("root")

            def mainloop(self):
                log.append("mainloop")

        def fake_app(root):
            log.append(("app", root))

        monkeypatch.setattr(gui.tk, "Tk", FakeRoot)
        monkeypatch.setattr(gui, "BlackScholesGUI", fake_app)
        return log

    def test_creates_a_root_builds_the_app_and_runs_the_mainloop(self, monkeypatch):
        log = self._patch_gui(monkeypatch)
        gui.run_gui()

        roots = [entry for entry in log if entry == "root"]
        mainloops = [entry for entry in log if entry == "mainloop"]
        apps = [entry for entry in log if isinstance(entry, tuple)]

        assert roots == ["root"]
        assert mainloops == ["mainloop"]
        assert len(apps) == 1
        assert apps[0][1] is not None

    def test_the_mainloop_runs_exactly_once(self, monkeypatch):
        log = self._patch_gui(monkeypatch)
        gui.run_gui()
        assert log.count("mainloop") == 1

    def test_the_app_is_built_before_the_mainloop_starts(self, monkeypatch):
        log = self._patch_gui(monkeypatch)
        gui.run_gui()
        assert log.index("root") < 1
        assert log[-1] == "mainloop"

    def test_the_app_receives_the_new_root(self, monkeypatch):
        created = []

        class FakeRoot:
            def __init__(self):
                created.append(self)

            def mainloop(self):
                pass

        passed = []
        monkeypatch.setattr(gui.tk, "Tk", FakeRoot)
        monkeypatch.setattr(gui, "BlackScholesGUI", passed.append)
        gui.run_gui()

        assert len(created) == 1
        assert passed == created

    def test_returns_none(self, monkeypatch):
        self._patch_gui(monkeypatch)
        assert gui.run_gui() is None


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
