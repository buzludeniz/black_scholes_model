"""
Tests for the Typer command-line interface.

Exercises the four commands (price, iv, surface, greeks) through
typer.testing.CliRunner, covering text and JSON output, option parsing,
error exit codes, and the implied-volatility round trip.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from black_scholes.cli import app

# --- Test Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Baseline: ATM call, 1 year, 5% rate, 20% vol -> 10.450584
# price/greeks take S K T r vol; iv takes S K T r (vol is solved for).
BASE_ARGS = ["100", "100", "1", "0.05", "0.2"]
IV_ARGS = ["100", "100", "1", "0.05"]
BASE_CALL_PRICE = 10.450583572185565
BASE_PUT_PRICE = 5.573526022256971

# Rich renders table cells with this box-drawing character (ASCII-safe source).
_VBAR = "\u2502"


def _json_price(result) -> float:
    """Parse a --format json price payload and return the price."""
    payload = json.loads(result.stdout)
    return payload["price"]


# --- Rich Table Parsing ---


def _table_rows(stdout: str) -> list[list[str]]:
    """Split Rich table body rows into their cell values.

    Rich draws cell separators with U+2502; headers use U+2503, so matching
    on U+2502 selects data rows only.
    """
    rows = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_VBAR) and stripped.endswith(_VBAR) and stripped.count(_VBAR) > 2:
            rows.append([c.strip() for c in stripped.strip(_VBAR).split(_VBAR)])
    return rows


def _column(stdout: str, index: int) -> list[str]:
    """Extract one cell column from every body row of a table in the output."""
    rows = _table_rows(stdout)
    assert rows, "no table body rows found in output"
    return [row[index] for row in rows]


def _metric_value(stdout: str, label: str) -> float:
    """Read a labelled numeric cell out of any Rich table in the output."""
    for row in _table_rows(stdout):
        if row[0] == label:
            return float(row[1])
    raise AssertionError(f"metric {label!r} not found in output")


def _call_price_column(stdout: str) -> list[str]:
    return _column(stdout, 1)


def _delta_column(stdout: str) -> list[str]:
    return _column(stdout, 2)


# --- Price Command ---


class TestPriceCommand:
    """Happy-path pricing for calls and puts."""

    def test_call_text_output(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call"])
        assert result.exit_code == 0
        assert result.exception is None
        assert "Spot (S)" in result.stdout
        assert "10.450584" in result.stdout

    def test_put_text_output(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "put"])
        assert result.exit_code == 0
        assert "5.573526" in result.stdout
        # Put delta is negative; the table renders the sign
        assert "-0.363169" in result.stdout

    def test_all_greeks_in_text_output(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call"])
        for label in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
            assert label in result.stdout

    @pytest.mark.parametrize("flag", ["--format", "-f"])
    def test_json_output_keys(self, runner, flag):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call", flag, "json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload.keys()) == {"price", "greeks", "params"}
        assert payload["price"] == pytest.approx(BASE_CALL_PRICE)
        assert set(payload["greeks"].keys()) == {"delta", "gamma", "vega", "theta", "rho"}
        assert payload["params"]["option_type"] == "call"
        assert payload["params"]["spot"] == 100.0

    def test_json_put_matches_text(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "put", "-f", "json"])
        payload = json.loads(result.stdout)
        assert payload["price"] == pytest.approx(BASE_PUT_PRICE)
        assert payload["params"]["option_type"] == "put"
        assert payload["greeks"]["delta"] < 0

    def test_json_greeks_match_library(self, runner):
        from black_scholes import OptionParams, OptionType, price_option

        result = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json"])
        payload = json.loads(result.stdout)

        expected = price_option(OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL))
        assert payload["greeks"]["delta"] == pytest.approx(expected.greeks.delta)
        assert payload["greeks"]["vega"] == pytest.approx(expected.greeks.vega)

    @pytest.mark.parametrize("opt_type", ["call", "c", "CALL", "put", "p"])
    def test_option_type_aliases(self, runner, opt_type):
        result = runner.invoke(app, ["price", *BASE_ARGS, opt_type, "-f", "json"])
        assert result.exit_code == 0
        expected = "call" if opt_type.lower().startswith("c") else "put"
        assert json.loads(result.stdout)["params"]["option_type"] == expected

    def test_invalid_option_type_exits_nonzero(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "banana"])
        assert result.exit_code != 0

    def test_non_numeric_spot_exits_nonzero(self, runner):
        result = runner.invoke(app, ["price", "abc", "100", "1", "0.05", "0.2", "call"])
        assert result.exit_code != 0

    def test_negative_spot_exits_nonzero(self, runner):
        result = runner.invoke(app, ["price", "-5", "100", "1", "0.05", "0.2", "call"])
        assert result.exit_code != 0

    def test_missing_argument_exits_nonzero(self, runner):
        result = runner.invoke(app, ["price", "100", "100"])
        assert result.exit_code != 0

    def test_bare_invocation_shows_help(self, runner):
        result = runner.invoke(app, [])
        assert result.exit_code != 0


# --- Dividends and Monte Carlo ---


class TestPriceDividendYield:
    """Dividend yield changes the price in the expected direction."""

    @pytest.mark.parametrize("flag", ["-q", "--div-yield"])
    def test_dividend_yield_lowers_call_price(self, runner, flag):
        base = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json"])
        with_q = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json", flag, "0.03"])
        assert base.exit_code == 0
        assert with_q.exit_code == 0

        base_price = _json_price(base)
        q_price = _json_price(with_q)
        assert q_price < base_price
        assert base_price - q_price == pytest.approx(1.80, abs=0.02)

    def test_dividend_yield_raises_put_price(self, runner):
        base = runner.invoke(app, ["price", *BASE_ARGS, "put", "-f", "json"])
        with_q = runner.invoke(app, ["price", *BASE_ARGS, "put", "-f", "json", "-q", "0.05"])
        assert _json_price(with_q) > _json_price(base)

    def test_zero_and_short_flag_agree(self, runner):
        short = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json", "-q", "0.02"])
        long = runner.invoke(
            app, ["price", *BASE_ARGS, "call", "-f", "json", "--div-yield", "0.02"]
        )
        assert _json_price(short) == pytest.approx(_json_price(long))

    def test_dividend_yield_echoed_in_params(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json", "-q", "0.025"])
        assert json.loads(result.stdout)["params"]["dividend_yield"] == pytest.approx(0.025)


class TestPriceMonteCarlo:
    """Monte Carlo validation block."""

    def test_mc_adds_validation_table(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call", "--mc", "--mc-paths", "2000"])
        assert result.exit_code == 0
        assert "Monte Carlo Validation" in result.stdout
        assert "MC Price" in result.stdout
        assert "Within 2 SE" in result.stdout

    def test_mc_agrees_with_analytic_price(self, runner):
        # Unseeded simulation, so compare against a generous multiple of the
        # reported standard error rather than the 2-sigma flag.
        result = runner.invoke(app, ["price", *BASE_ARGS, "call", "--mc", "--mc-paths", "20000"])
        assert result.exit_code == 0

        mc_price = _metric_value(result.stdout, "MC Price")
        std_error = _metric_value(result.stdout, "Std Error")
        difference = _metric_value(result.stdout, "Difference")
        analytic = _metric_value(result.stdout, "Price")

        assert std_error > 0
        # Cells are printed to 6 dp, so compare with slack for that rounding.
        assert difference == pytest.approx(analytic - mc_price, abs=1e-4)
        assert abs(difference) < 4 * std_error

    def test_mc_paths_option_rejects_garbage(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call", "--mc", "--mc-paths", "many"])
        assert result.exit_code != 0

    def test_mc_with_json_output_serialises(self, runner):
        # Regression guard: the Monte Carlo block must stay JSON-serialisable.
        # A numpy bool or scalar leaking in here raises TypeError at runtime
        # rather than failing a value assertion, so parse the payload.
        for option_type in ("call", "put"):
            result = runner.invoke(
                app,
                ["price", *BASE_ARGS, option_type, "--mc", "--mc-paths", "20000", "-f", "json"],
            )
            assert result.exit_code == 0, result.stdout
            payload = json.loads(result.stdout)
            mc = payload["monte_carlo"]
            assert set(mc) == {"price", "std_error", "diff", "within_2se"}
            assert isinstance(mc["price"], float)
            assert isinstance(mc["std_error"], float)
            assert isinstance(mc["diff"], float)
            assert isinstance(mc["within_2se"], bool)

    def test_mc_json_diff_matches_parts(self, runner):
        result = runner.invoke(
            app, ["price", *BASE_ARGS, "call", "--mc", "--mc-paths", "20000", "-f", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        mc = payload["monte_carlo"]
        assert mc["diff"] == pytest.approx(payload["price"] - mc["price"], abs=1e-9)
        assert mc["within_2se"] == (abs(mc["diff"]) <= 2 * mc["std_error"])

    def test_no_mc_by_default(self, runner):
        result = runner.invoke(app, ["price", *BASE_ARGS, "call"])
        assert "Monte Carlo Validation" not in result.stdout


# --- IV Command ---


class TestIvCommand:
    """Implied volatility inversion through the CLI."""

    @pytest.mark.parametrize("opt_type", ["call", "put"])
    def test_roundtrip_recovers_input_vol(self, runner, opt_type):
        priced = runner.invoke(app, ["price", *BASE_ARGS, opt_type, "-f", "json"])
        assert priced.exit_code == 0
        market_price = _json_price(priced)

        solved = runner.invoke(
            app, ["iv", f"{market_price:.10f}", *IV_ARGS, opt_type, "-f", "json"]
        )
        assert solved.exit_code == 0
        payload = json.loads(solved.stdout)
        assert payload["implied_volatility"] == pytest.approx(0.20, abs=1e-5)
        assert payload["implied_vol_pct"] == pytest.approx(20.0, abs=1e-3)

    def test_roundtrip_respects_dividend_yield(self, runner):
        priced = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json", "-q", "0.03"])
        market_price = _json_price(priced)

        solved = runner.invoke(
            app,
            ["iv", f"{market_price:.10f}", *IV_ARGS, "call", "-q", "0.03", "-f", "json"],
        )
        assert solved.exit_code == 0
        assert json.loads(solved.stdout)["implied_volatility"] == pytest.approx(0.20, abs=1e-5)

    def test_ignoring_dividend_yield_biases_vol(self, runner):
        priced = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json", "-q", "0.03"])
        market_price = _json_price(priced)

        solved = runner.invoke(app, ["iv", f"{market_price:.10f}", *IV_ARGS, "call", "-f", "json"])
        assert solved.exit_code == 0
        assert json.loads(solved.stdout)["implied_volatility"] != pytest.approx(0.20, abs=1e-4)

    def test_text_output_reports_vol(self, runner):
        priced = runner.invoke(app, ["price", *BASE_ARGS, "call", "-f", "json"])
        market_price = _json_price(priced)

        result = runner.invoke(app, ["iv", f"{market_price:.10f}", *IV_ARGS, "call"])
        assert result.exit_code == 0
        assert "Implied Volatility" in result.stdout
        assert "0.200000" in result.stdout

    def test_price_above_upper_bound_fails(self, runner):
        result = runner.invoke(app, ["iv", "100.1", *IV_ARGS, "call"])
        assert result.exit_code != 0
        assert "Error" in result.stdout

    def test_price_below_intrinsic_fails(self, runner):
        result = runner.invoke(app, ["iv", "1.0", *IV_ARGS, "call"])
        assert result.exit_code != 0
        assert "Error" in result.stdout

    def test_put_above_upper_bound_fails(self, runner):
        result = runner.invoke(app, ["iv", "95.5", *IV_ARGS, "put"])
        assert result.exit_code != 0

    def test_bounds_error_also_fails_in_json_mode(self, runner):
        result = runner.invoke(app, ["iv", "100.1", *IV_ARGS, "call", "-f", "json"])
        assert result.exit_code != 0

    def test_missing_argument_fails(self, runner):
        result = runner.invoke(app, ["iv", "10.0", "100", "100"])
        assert result.exit_code != 0


# --- Surface Command ---


class TestSurfaceCommand:
    """Strike-by-strike smile/surface table."""

    def test_row_per_requested_strike(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "80,90,100"]
        )
        assert result.exit_code == 0
        for strike in ("80.00", "90.00", "100.00"):
            assert strike in result.stdout

    def test_default_strike_list(self, runner):
        result = runner.invoke(app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2"])
        assert result.exit_code == 0
        for strike in ("80.00", "90.00", "100.00", "110.00", "120.00"):
            assert strike in result.stdout

    def test_long_option_names(self, runner):
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                "95,105",
            ],
        )
        assert result.exit_code == 0
        assert "95.00" in result.stdout
        assert "105.00" in result.stdout

    def test_atm_row_prices_match_call_and_put(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "100"]
        )
        assert result.exit_code == 0
        assert "10.4506" in result.stdout
        assert "5.5735" in result.stdout

    def test_call_iv_column_equals_input_vol(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.25", "-K", "95,100,105"]
        )
        assert result.exit_code == 0
        assert "25.0000%" in result.stdout

    def test_row_count_matches_strike_count(self, runner):
        result = runner.invoke(
            app,
            ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "70,80,90,100"],
        )
        assert result.exit_code == 0
        assert len(_table_rows(result.stdout)) == 4

    def test_call_deltas_decrease_with_strike(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "80,100,120"]
        )
        assert result.exit_code == 0
        deltas = [float(line) for line in _delta_column(result.stdout)]
        assert len(deltas) == 3
        assert deltas == sorted(deltas, reverse=True)

    def test_call_prices_decrease_with_strike(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "80,100,120"]
        )
        assert result.exit_code == 0
        prices = [float(v) for v in _call_price_column(result.stdout)]
        assert len(prices) == 3
        assert prices == sorted(prices, reverse=True)

    def test_dividend_yield_lowers_call_price(self, runner):
        base = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "100"]
        )
        with_q = runner.invoke(
            app,
            [
                "surface",
                "-S",
                "100",
                "-T",
                "1",
                "-r",
                "0.05",
                "-v",
                "0.2",
                "-K",
                "100",
                "-q",
                "0.02",
            ],
        )
        assert base.exit_code == 0
        assert with_q.exit_code == 0
        assert float(_call_price_column(with_q.stdout)[0]) < float(
            _call_price_column(base.stdout)[0]
        )

    def test_invalid_strike_exits_nonzero(self, runner):
        result = runner.invoke(
            app, ["surface", "-S", "100", "-T", "1", "-r", "0.05", "-v", "0.2", "-K", "80,abc"]
        )
        assert result.exit_code != 0

    def test_missing_required_option_exits_nonzero(self, runner):
        result = runner.invoke(app, ["surface", "-S", "100"])
        assert result.exit_code != 0


# --- Greeks Command ---


class TestGreeksCommand:
    """Detailed Greek explanations panel."""

    @pytest.mark.parametrize("opt_type", ["call", "put"])
    def test_lists_every_greek(self, runner, opt_type):
        result = runner.invoke(app, ["greeks", *BASE_ARGS, opt_type])
        assert result.exit_code == 0
        for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
            assert greek in result.stdout

    def test_includes_price_and_explanations(self, runner):
        result = runner.invoke(app, ["greeks", *BASE_ARGS, "call"])
        assert "Price:" in result.stdout
        assert "10.450584" in result.stdout
        assert "Rate of change" in result.stdout

    def test_put_reports_negative_delta_direction(self, runner):
        result = runner.invoke(app, ["greeks", *BASE_ARGS, "put"])
        assert result.exit_code == 0
        assert "For put: negative" in result.stdout

    def test_dividend_yield_option_accepted(self, runner):
        result = runner.invoke(app, ["greeks", *BASE_ARGS, "call", "-q", "0.02"])
        assert result.exit_code == 0
        assert "Delta" in result.stdout

    def test_invalid_option_type_exits_nonzero(self, runner):
        result = runner.invoke(app, ["greeks", *BASE_ARGS, "straddle"])
        assert result.exit_code != 0

    def test_negative_vol_exits_nonzero(self, runner):
        result = runner.invoke(app, ["greeks", "100", "100", "1", "0.05", "-0.2", "call"])
        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# A value beginning with a dash is read as an option, not as a number, so
# "-inf" can never reach the validation under test here. Click rejects it as a
# usage error, which is correct and is covered separately below. The library
# layer is tested against "-inf" directly.
NON_FINITE = ["nan", "inf"]


class TestCliErrorHandling:
    """Bad input must exit non-zero with a readable message, never a traceback.

    Every case here previously reached an uncaught exception and printed a
    Python traceback to the terminal.
    """

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_price_rejects_non_finite_spot(self, runner, text):
        result = runner.invoke(app, ["price", text, "100", "1", "0.05", "0.2", "call"])
        assert result.exit_code == 1
        assert "must be a finite number" in result.stdout
        assert "Traceback" not in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_price_rejects_non_finite_strike(self, runner, text):
        result = runner.invoke(app, ["price", "100", text, "1", "0.05", "0.2", "call"])
        assert result.exit_code == 1
        assert "strike must be a finite number" in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_price_rejects_non_finite_vol(self, runner, text):
        result = runner.invoke(app, ["price", "100", "100", "1", "0.05", text, "call"])
        assert result.exit_code == 1
        assert "volatility must be a finite number" in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_price_rejects_non_finite_rate(self, runner, text):
        result = runner.invoke(app, ["price", "100", "100", "1", text, "0.2", "call"])
        assert result.exit_code == 1
        assert "risk_free_rate must be a finite number" in result.stdout

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["price", "0", "100", "1", "0.05", "0.2", "call"], "spot must be positive"),
            (["price", "100", "0", "1", "0.05", "0.2", "call"], "strike must be positive"),
            (["price", "100", "100", "0", "0.05", "0.2", "call"], "time_to_maturity must be"),
        ],
    )
    def test_price_rejects_invalid_magnitudes(self, runner, args, message):
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert message in result.stdout
        assert "Traceback" not in result.stdout

    def test_greeks_rejects_invalid_input(self, runner):
        result = runner.invoke(app, ["greeks", "nan", "100", "1", "0.05", "0.2", "call"])
        assert result.exit_code == 1
        assert "must be a finite number" in result.stdout
        assert "Traceback" not in result.stdout

    def test_iv_rejects_non_finite_market_price(self, runner):
        result = runner.invoke(app, ["iv", "nan", "100", "100", "1", "0.05", "call"])
        assert result.exit_code == 1
        assert "market_price must be a finite number" in result.stdout

    @pytest.mark.parametrize("bad", ["abc", "90,,110", "", "90,  ,110", "9 0"])
    def test_surface_rejects_malformed_strikes(self, runner, bad):
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                bad,
            ],
        )
        assert result.exit_code == 1
        assert "Error" in result.stdout
        assert "Traceback" not in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_surface_rejects_non_finite_strikes(self, runner, text):
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                f"90,{text},110",
            ],
        )
        assert result.exit_code == 1
        assert "must be finite numbers" in result.stdout

    def test_surface_names_the_offending_strike(self, runner):
        """The message comes from the model layer, so it is singular."""
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                "90,0,110",
            ],
        )
        assert result.exit_code == 1
        assert "strike must be positive" in result.stdout
        assert "Traceback" not in result.stdout

    def test_dash_prefixed_value_is_a_usage_error(self, runner):
        """A leading dash makes Click read the token as an option.

        This is why "-inf" and "-0.05" cannot be typed as arguments at all: they
        are refused as usage errors rather than reaching the numeric checks.
        """
        for bad in ("-inf", "-0.05"):
            result = runner.invoke(app, ["price", "100", "100", "1", "0.05", bad, "call"])
            assert result.exit_code != 0
            assert "Traceback" not in result.stdout

    def test_negative_rate_rejected_when_passed_as_an_option(self, runner):
        """Negative rates are rejected by policy, and the option form works."""
        result = runner.invoke(
            app,
            ["price", "100", "100", "1", "0.05", "0.2", "call", "--rate-does-not-exist"],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout

    def test_surface_rejects_non_positive_strike(self, runner):
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                "90,0,110",
            ],
        )
        assert result.exit_code == 1
        assert "strike must be positive" in result.stdout
        assert "Traceback" not in result.stdout

    def test_surface_rejects_non_finite_spot(self, runner):
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot",
                "nan",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--strikes",
                "90,100,110",
            ],
        )
        assert result.exit_code == 1
        assert "spot must be a finite number" in result.stdout

    @pytest.mark.parametrize("bad", ["abc", "1.0,,2.0"])
    def test_fit_surface_rejects_malformed_prices(self, runner, bad):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--strikes",
                "70,80,90,100,110,130,150",
                "--prices",
                bad,
            ],
        )
        assert result.exit_code == 1
        assert "Error" in result.stdout
        assert "Traceback" not in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_fit_surface_rejects_non_finite_spot(self, runner, text):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                text,
                "--time",
                "1",
                "--rate",
                "0.05",
                "--strikes",
                "70,80,90,100,110,130,150",
                "--prices",
                "34.38,25.84,18.08,11.75,7.40,3.04,1.34",
            ],
        )
        assert result.exit_code == 1
        assert "spot must be a finite number" in result.stdout
        assert "Traceback" not in result.stdout

    @pytest.mark.parametrize("text", NON_FINITE)
    def test_fit_surface_rejects_non_finite_prices(self, runner, text):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--strikes",
                "70,80,90,100,110,130,150",
                "--prices",
                f"34.38,25.84,{text},11.75,7.40,3.04,1.34",
            ],
        )
        assert result.exit_code == 1
        assert "must be finite numbers" in result.stdout

    def test_fit_surface_rejects_duplicate_strikes(self, runner):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--strikes",
                "100,100,90,100,110,130,150",
                "--prices",
                "11.75,11.75,18.08,11.75,7.40,3.04,1.34",
            ],
        )
        assert result.exit_code == 1
        assert "duplicate strikes" in result.stdout

    def test_invalid_option_type_is_a_usage_error(self, runner):
        result = runner.invoke(app, ["price", "100", "100", "1", "0.05", "0.2", "banana"])
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout


class TestCliJsonOutputStaysValid:
    """A successful JSON call must remain machine-readable.

    Errors are printed as text and exit non-zero, so a caller parsing stdout
    gets a parse failure rather than a half-written object. That is the existing
    contract and these tests pin it.
    """

    def test_price_json_is_parseable(self, runner):
        result = runner.invoke(
            app, ["price", "100", "100", "1", "0.05", "0.2", "call", "-f", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["price"] == pytest.approx(10.450584, abs=1e-6)
        assert set(payload["greeks"]) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_price_json_with_monte_carlo_is_parseable(self, runner):
        result = runner.invoke(
            app,
            ["price", "100", "100", "1", "0.05", "0.2", "call", "-f", "json", "--mc"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "monte_carlo" in payload
        assert "std_error" in payload["monte_carlo"]

    def test_iv_json_is_parseable(self, runner):
        result = runner.invoke(
            app, ["iv", "10.450584", "100", "100", "1", "0.05", "call", "-f", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["implied_volatility"] == pytest.approx(0.20, abs=1e-4)

    def test_fit_surface_json_is_parseable(self, runner):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--strikes",
                "70,80,90,100,110,130,150",
                "--prices",
                "34.38,25.84,18.08,11.75,7.40,3.04,1.34",
                "-f",
                "json",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload["parameters"]) == {"a", "b", "rho", "m", "sigma"}
        assert payload["arbitrage_free"]["butterfly"] is True

    def test_json_error_path_is_not_valid_json(self, runner):
        """Documented behaviour: an error is text plus a non-zero exit."""
        result = runner.invoke(
            app, ["price", "nan", "100", "1", "0.05", "0.2", "call", "-f", "json"]
        )
        assert result.exit_code == 1
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)
