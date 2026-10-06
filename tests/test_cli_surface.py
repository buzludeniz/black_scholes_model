"""
Tests for the fit-surface CLI command.

The command inverts market prices, fits an SVI smile, and reports the fitted
parameters with arbitrage status.
"""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from black_scholes import OptionParams, OptionType, black_scholes_price
from black_scholes.cli import app
from black_scholes.surface import SVICurve

# Ground-truth curve and the prices it implies.
TRUTH = SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0]
SPOT = 100.0
RATE = 0.05
MATURITY = 1.0


def _prices(option_type: OptionType = OptionType.CALL, q: float = 0.0) -> list[float]:
    out = []
    for strike in STRIKES:
        vol = TRUTH.implied_volatility(math.log(strike / SPOT), MATURITY)
        out.append(
            black_scholes_price(OptionParams(SPOT, strike, MATURITY, RATE, vol, option_type, q))
        )
    return out


ARGS = [
    "--spot",
    str(SPOT),
    "--strikes",
    ",".join(str(k) for k in STRIKES),
    "--time",
    str(MATURITY),
    "--rate",
    str(RATE),
]


def _with_prices(prices: list[float], *extra: str) -> list[str]:
    return [*ARGS, "--prices", ",".join(f"{p:.6f}" for p in prices), *extra]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestFitSurfaceSuccess:
    """A clean arbitrage-free smile."""

    def test_reports_parameters(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices())])
        assert result.exit_code == 0, result.stdout
        for label in ("a     =", "b     =", "rho   =", "m     =", "sigma ="):
            assert label in result.stdout

    def test_recovers_known_parameters(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        assert result.exit_code == 0
        params = json.loads(result.stdout)["parameters"]
        assert params["a"] == pytest.approx(TRUTH.a, abs=1e-4)
        assert params["b"] == pytest.approx(TRUTH.b, abs=1e-4)
        assert params["rho"] == pytest.approx(TRUTH.rho, abs=1e-4)
        assert params["sigma"] == pytest.approx(TRUTH.sigma, abs=1e-4)

    def test_json_shape(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload) == {
            "forward",
            "parameters",
            "rms_error",
            "max_error",
            "arbitrage_free",
            "weights",
            "strikes",
            "market_vols",
            "model_vols",
        }
        assert payload["arbitrage_free"] == {"parameters": True, "butterfly": True}

    def test_json_numbers_are_native(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        payload = json.loads(result.stdout)
        for key in ("a", "b", "rho", "m", "sigma"):
            assert type(payload["parameters"][key]) is float, key
        for key in ("rms_error", "max_error"):
            assert type(payload[key]) is float, key

    def test_model_vols_match_market_vols(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        payload = json.loads(result.stdout)
        assert payload["max_error"] < 1e-4
        assert payload["rms_error"] < 1e-4

    def test_row_per_strike(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        payload = json.loads(result.stdout)
        assert len(payload["strikes"]) == len(STRIKES)
        assert len(payload["model_vols"]) == len(STRIKES)

    def test_butterfly_reported_free(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices())])
        assert "Butterfly  : free" in result.stdout

    def test_put_prices_accepted(self, runner):
        result = runner.invoke(
            app, ["fit-surface", *_with_prices(_prices(OptionType.PUT)), "--type", "put"]
        )
        assert result.exit_code == 0, result.stdout
        assert "free" in result.stdout

    def test_dividend_yield_accepted(self, runner):
        prices = _prices(q=0.03)
        result = runner.invoke(app, ["fit-surface", *_with_prices(prices), "-q", "0.03"])
        assert result.exit_code == 0, result.stdout

    def test_short_flag_for_format(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "-f", "json"])
        assert result.exit_code == 0


class TestFitSurfaceFailures:
    """Input the command must reject."""

    def test_arbitrage_violating_price_exits_nonzero(self, runner):
        """A price below intrinsic is rejected during inversion, before fitting."""
        bad = [1.0, 18.0, 18.08, 11.75, 7.4, 3.0, 1.34]
        result = runner.invoke(app, ["fit-surface", *_with_prices(bad)])
        assert result.exit_code != 0
        assert "no-arbitrage bounds" in result.stdout

    def test_fit_rejection_suggests_relaxed(self, runner):
        """A rejection that --relaxed can actually bypass must say so.

        The hint is reserved for fit failures. A quote that fails inversion is
        rejected with or without --relaxed, so offering it there would point at
        a flag that cannot help.
        """
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0]
        vols = [0.20, 0.22, 0.40, 0.60, 0.40, 0.22, 0.20]
        prices = [
            black_scholes_price(OptionParams(SPOT, k, MATURITY, RATE, v, OptionType.CALL))
            for k, v in zip(strikes, vols, strict=True)
        ]
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                str(SPOT),
                "--time",
                str(MATURITY),
                "--rate",
                str(RATE),
                "--strikes",
                ",".join(f"{k:g}" for k in strikes),
                "--prices",
                ",".join(f"{p:.6f}" for p in prices),
            ],
        )
        assert result.exit_code != 0
        assert "--relaxed" in result.stdout

    def test_inversion_failure_does_not_suggest_relaxed(self, runner):
        """A price outside the arbitrage bounds is not fixable by --relaxed."""
        bad = [1.0, 18.0, 18.08, 11.75, 7.4, 3.0, 1.34]
        result = runner.invoke(app, ["fit-surface", *_with_prices(bad)])
        assert result.exit_code != 0
        assert "--relaxed" not in result.stdout

    def test_unfittable_shape_exits_nonzero(self, runner):
        """Prices that invert fine but whose smile no SVI curve can match.

        These are individually arbitrage-valid, so the failure comes from the
        error tolerance rather than from price inversion. That is the guard
        stopping a 25-volatility-point miss being reported as a good fit.
        """
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0]
        vols = [0.20, 0.22, 0.40, 0.60, 0.40, 0.22, 0.20]
        prices = [
            black_scholes_price(OptionParams(SPOT, k, MATURITY, RATE, v, OptionType.CALL))
            for k, v in zip(strikes, vols, strict=True)
        ]
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                str(SPOT),
                "--strikes",
                ",".join(str(k) for k in strikes),
                "--prices",
                ",".join(f"{p:.6f}" for p in prices),
                "--time",
                str(MATURITY),
                "--rate",
                str(RATE),
            ],
        )
        assert result.exit_code != 0
        assert "cannot fit this smile" in result.stdout

    def test_relaxed_flag_bypasses_the_fit_guards(self, runner):
        """--relaxed lifts the error bound so the fit can be inspected."""
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0]
        vols = [0.20, 0.22, 0.40, 0.60, 0.40, 0.22, 0.20]
        prices = [
            black_scholes_price(OptionParams(SPOT, k, MATURITY, RATE, v, OptionType.CALL))
            for k, v in zip(strikes, vols, strict=True)
        ]
        base = [
            "fit-surface",
            "--spot",
            str(SPOT),
            "--strikes",
            ",".join(str(k) for k in strikes),
            "--prices",
            ",".join(f"{p:.6f}" for p in prices),
            "--time",
            str(MATURITY),
            "--rate",
            str(RATE),
        ]
        strict = runner.invoke(app, base)
        assert strict.exit_code != 0

        relaxed = runner.invoke(app, [*base, "--relaxed", "-f", "json"])
        assert relaxed.exit_code == 0, relaxed.stdout
        payload = json.loads(relaxed.stdout)
        assert payload["max_error"] > 0.2, "the guard was hiding a poor fit"
        # This is precisely why an error bound is needed alongside the
        # butterfly check: the fit that slipped through is arbitrage-free, so
        # arbitrage status alone would never have caught it.
        assert payload["arbitrage_free"]["butterfly"] is True
        assert payload["arbitrage_free"]["parameters"] is True

    def test_too_few_strikes_exits_nonzero(self, runner):
        strikes = STRIKES[:3]
        prices = [
            black_scholes_price(OptionParams(SPOT, k, MATURITY, RATE, 0.22, OptionType.CALL))
            for k in strikes
        ]
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                str(SPOT),
                "--strikes",
                ",".join(str(k) for k in strikes),
                "--prices",
                ",".join(f"{p:.6f}" for p in prices),
                "--time",
                str(MATURITY),
                "--rate",
                str(RATE),
            ],
        )
        assert result.exit_code != 0
        assert "at least 4 strikes" in result.stdout

    def test_length_mismatch_exits_nonzero(self, runner):
        prices = ",".join("18.0" for _ in STRIKES[:-1])
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--strikes",
                ",".join(str(k) for k in STRIKES),
                "--prices",
                prices,
                "--time",
                "1",
                "--rate",
                "0.05",
            ],
        )
        assert result.exit_code != 0
        assert "differ in length" in result.stdout

    def test_non_numeric_strikes_exits_nonzero(self, runner):
        result = runner.invoke(
            app,
            [
                "fit-surface",
                "--spot",
                "100",
                "--strikes",
                "70,banana,90,100,110",
                "--prices",
                "1,2,3,4,5",
                "--time",
                "1",
                "--rate",
                "0.05",
            ],
        )
        assert result.exit_code != 0
        assert "comma-separated numbers" in result.stdout

    def test_invalid_option_type_exits_nonzero(self, runner):
        result = runner.invoke(app, ["fit-surface", *_with_prices(_prices()), "--type", "banana"])
        assert result.exit_code != 0


class TestFitSurfaceRegistration:
    """The command is reachable under the expected name."""

    def test_listed_in_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fit-surface" in result.stdout

    def test_command_help_works(self, runner):
        result = runner.invoke(app, ["fit-surface", "--help"])
        assert result.exit_code == 0
        assert "SVI" in result.stdout
