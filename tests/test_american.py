"""
Tests for American option pricing: the Cox-Ross-Rubinstein binomial tree and
the Barone-Adesi-Whaley (BAW) quadratic approximation.

Reference values and where they come from:

- J. C. Hull, *Options, Futures, and Other Derivatives* (American put example,
  S = K = 50, r = 10%, sigma = 40%, T = 5 months): 4.49 on a 5-step CRR tree,
  and 4.263 / 4.272 / 4.278 / 4.283 with 30 / 50 / 100 / 500 steps.
- E. G. Haug, *The Complete Guide to Option Pricing Formulas* (2nd ed., 2007),
  Barone-Adesi and Whaley section: the worked American put on a future
  (S = K = 100, T = 0.5, r = 10%, b = 0, sigma = 25%), 6.801362, also used as
  the example in the R package fOptions, which ports Haug's code.
- Haug's comparison table of BAW American call values for K = 100, r = 10%,
  b = 0 (a futures option, so here q = r), T in {0.1, 0.5},
  sigma in {15%, 25%, 35%}, S in {90, 100, 110}. Checked to 0.003, the
  largest difference from the transcribed table (most entries agree to all four
  printed decimals; the few that do not sit next to the exercise boundary).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from black_scholes import OptionParams, OptionType, black_scholes_price
from black_scholes.american import (
    DEFAULT_STEPS,
    american_price,
    barone_adesi_whaley_price,
    binomial_price,
)

CALL, PUT = OptionType.CALL, OptionType.PUT


def opt(spot=100.0, strike=100.0, t=1.0, r=0.05, vol=0.20, kind=PUT, q=0.0) -> OptionParams:
    return OptionParams(spot, strike, t, r, vol, kind, q)


def intrinsic(p: OptionParams) -> float:
    if p.option_type is CALL:
        return max(p.spot - p.strike, 0.0)
    return max(p.strike - p.spot, 0.0)


GRID = [
    opt(spot=s, t=t, r=r, vol=v, kind=k, q=q)
    for s, t, r, q, v, k in itertools.product(
        (80.0, 100.0, 120.0), (0.25, 1.0), (0.0, 0.06), (0.0, 0.08), (0.15, 0.40), (CALL, PUT)
    )
]


class TestReferenceValues:
    """Published numbers, each cited in the module docstring."""

    HULL = opt(spot=50.0, strike=50.0, t=5 / 12, r=0.10, vol=0.40)

    def test_hull_five_step_tree(self):
        assert binomial_price(self.HULL, 5) == pytest.approx(4.49, abs=5e-3)

    @pytest.mark.parametrize(
        ("steps", "expected"), [(30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)]
    )
    def test_hull_tree_convergence_table(self, steps, expected):
        assert binomial_price(self.HULL, steps) == pytest.approx(expected, abs=5e-4)

    def test_haug_baw_put_on_future(self):
        put = opt(spot=100.0, strike=100.0, t=0.5, r=0.10, vol=0.25, kind=PUT, q=0.10)
        assert barone_adesi_whaley_price(put) == pytest.approx(6.801362, abs=1e-4)

    # (T, sigma) -> BAW call values at S = 90, 100, 110; K = 100, r = 10%, b = 0.
    HAUG_CALLS: dict[tuple[float, float], tuple[float, float, float]] = {  # noqa: RUF012
        (0.1, 0.15): (0.0206, 1.8771, 10.0089),
        (0.1, 0.25): (0.3159, 3.1280, 10.3919),
        (0.1, 0.35): (0.9495, 4.3777, 11.1679),
        (0.5, 0.15): (0.8208, 4.0842, 10.8087),
        (0.5, 0.25): (2.7437, 6.8015, 13.0170),
        (0.5, 0.35): (5.0063, 9.5106, 15.5689),
    }

    @pytest.mark.parametrize(("t", "vol"), list(HAUG_CALLS))
    def test_haug_baw_call_table(self, t, vol):
        for spot, expected in zip((90.0, 100.0, 110.0), self.HAUG_CALLS[(t, vol)], strict=True):
            call = opt(spot=spot, t=t, r=0.10, vol=vol, kind=CALL, q=0.10)
            assert barone_adesi_whaley_price(call) == pytest.approx(expected, abs=3e-3)

    def test_haug_table_is_close_to_the_tree(self):
        """The same table, against the converged tree: BAW is an approximation."""
        for (t, vol), row in self.HAUG_CALLS.items():
            for spot, expected in zip((90.0, 100.0, 110.0), row, strict=True):
                tree = binomial_price(opt(spot=spot, t=t, r=0.10, vol=vol, kind=CALL, q=0.10), 2000)
                assert tree == pytest.approx(expected, abs=0.05)


class TestNoDividendCall:
    """Merton (1973): without dividends a call is never exercised early."""

    @pytest.mark.parametrize("spot", [70.0, 90.0, 100.0, 110.0, 150.0])
    @pytest.mark.parametrize("t", [0.1, 1.0, 3.0])
    def test_baw_equals_european(self, spot, t):
        call = opt(spot=spot, t=t, r=0.07, vol=0.3, kind=CALL)
        assert barone_adesi_whaley_price(call) == black_scholes_price(call)

    @pytest.mark.parametrize("spot", [70.0, 100.0, 150.0])
    def test_tree_has_no_early_exercise(self, spot):
        call = opt(spot=spot, r=0.07, vol=0.3, kind=CALL)
        assert binomial_price(call, 400) == pytest.approx(
            binomial_price(call, 400, american=False), abs=1e-12
        )

    @pytest.mark.parametrize("spot", [70.0, 100.0, 150.0])
    def test_tree_equals_european_closed_form(self, spot):
        call = opt(spot=spot, r=0.07, vol=0.3, kind=CALL)
        assert binomial_price(call, 2000) == pytest.approx(black_scholes_price(call), abs=5e-3)

    def test_american_price_wrapper(self):
        call = opt(kind=CALL)
        assert american_price(call, method="baw") == black_scholes_price(call)
        assert american_price(call) == binomial_price(call, DEFAULT_STEPS)


class TestLowerBounds:
    """An American option is worth at least the European and its intrinsic value."""

    @pytest.mark.parametrize(
        "p",
        GRID,
        ids=lambda p: (
            f"{p.option_type.value}-{p.spot}-{p.time_to_maturity}-{p.risk_free_rate}-{p.dividend_yield}-{p.volatility}"
        ),
    )
    def test_at_least_european_and_intrinsic(self, p):
        euro = black_scholes_price(p)
        floor = max(euro, intrinsic(p))
        assert barone_adesi_whaley_price(p) >= floor - 1e-12
        tree = binomial_price(p, 500)
        # The tree carries its own discretisation error relative to the closed
        # form, so compare it to the European price on the same tree as well.
        assert tree >= binomial_price(p, 500, american=False) - 1e-12
        assert tree >= intrinsic(p) - 1e-12
        assert tree >= euro - 0.01

    def test_american_put_strictly_above_european_with_positive_rate(self):
        put = opt(r=0.08)
        assert binomial_price(put, 1000) > black_scholes_price(put) + 0.1
        assert barone_adesi_whaley_price(put) > black_scholes_price(put) + 0.1

    def test_dividend_call_has_early_exercise_premium(self):
        call = opt(spot=120.0, r=0.05, q=0.10, kind=CALL)
        assert binomial_price(call, 1000) > black_scholes_price(call) + 0.1
        assert barone_adesi_whaley_price(call) > black_scholes_price(call) + 0.1

    @pytest.mark.parametrize("spot", [40.0, 60.0])
    def test_deep_in_the_money_put_is_intrinsic(self, spot):
        put = opt(spot=spot, r=0.08)
        assert barone_adesi_whaley_price(put) == pytest.approx(100.0 - spot, abs=1e-12)
        assert binomial_price(put, 500) == pytest.approx(100.0 - spot, abs=1e-12)

    def test_deep_in_the_money_dividend_call_is_intrinsic(self):
        call = opt(spot=200.0, r=0.05, q=0.15, vol=0.2, kind=CALL)
        assert barone_adesi_whaley_price(call) == pytest.approx(100.0, abs=1e-12)

    def test_zero_rate_put_is_european(self):
        """With r = 0 early exercise of a put gains nothing."""
        put = opt(r=0.0, q=0.02)
        assert barone_adesi_whaley_price(put) == black_scholes_price(put)
        assert binomial_price(put, 500) == pytest.approx(
            binomial_price(put, 500, american=False), abs=1e-12
        )

    def test_zero_rate_dividend_call_uses_limit(self):
        """r = 0 with q > 0 takes the continuous r -> 0 limit of 4M/h."""
        call = opt(spot=110.0, r=0.0, q=0.06, vol=0.25, kind=CALL)
        near = opt(spot=110.0, r=1e-9, q=0.06, vol=0.25, kind=CALL)
        assert barone_adesi_whaley_price(call) == pytest.approx(
            barone_adesi_whaley_price(near), abs=1e-6
        )
        assert barone_adesi_whaley_price(call) == pytest.approx(
            binomial_price(call, 2000), abs=0.12
        )


class TestConvergence:
    def test_tree_error_shrinks_with_steps(self):
        """Doubling the steps roughly halves the error against a fine reference."""
        put = opt(r=0.06, vol=0.25)
        reference = binomial_price(put, 8000)
        errors = [abs(binomial_price(put, n) - reference) for n in (50, 100, 200, 400, 800)]
        assert all(later < earlier for earlier, later in itertools.pairwise(errors))
        assert errors[-1] < 0.003

    def test_successive_prices_settle(self):
        put = opt(r=0.06, vol=0.25)
        prices = [binomial_price(put, n) for n in (250, 500, 1000, 2000)]
        gaps = [abs(b - a) for a, b in itertools.pairwise(prices)]
        assert all(later < earlier for earlier, later in itertools.pairwise(gaps))

    def test_european_tree_converges_to_black_scholes(self):
        for kind in (CALL, PUT):
            p = opt(r=0.05, q=0.02, vol=0.3, kind=kind)
            exact = black_scholes_price(p)
            errors = [abs(binomial_price(p, n, american=False) - exact) for n in (100, 400, 1600)]
            assert errors[0] > errors[1] > errors[2]
            assert errors[2] < 2e-3


class TestBAWAgainstTree:
    """BAW is an approximation; the converged tree is the reference.

    Stated tolerance: ``|BAW - CRR(2000)| <= 0.12`` (absolute, on K = 100) for
    maturities up to one year, vols 15% to 40%, rates and yields 0 to 8%.
    Measured worst case on this grid: 0.117, a deep in-the-money call with
    r = 0, q = 8%, sigma = 40%, T = 1, where BAW underprices near the exercise
    boundary; the McDonald-Schroder-symmetric put shows the same gap, so it is
    a property of the approximation, not of one branch of the code. For short
    maturities and moderate volatility the two agree to about a cent.
    """

    @pytest.mark.parametrize(
        "p",
        GRID,
        ids=lambda p: (
            f"{p.option_type.value}-{p.spot}-{p.time_to_maturity}-{p.risk_free_rate}-{p.dividend_yield}-{p.volatility}"
        ),
    )
    def test_agree_within_tolerance(self, p):
        tree = binomial_price(p, 2000)
        assert barone_adesi_whaley_price(p) == pytest.approx(tree, abs=0.12)

    def test_short_dated_agreement_is_tight(self):
        for spot in (90.0, 100.0, 110.0):
            p = opt(spot=spot, t=0.1, r=0.08, vol=0.2)
            assert barone_adesi_whaley_price(p) == pytest.approx(binomial_price(p, 2000), abs=0.01)


class TestSymmetry:
    def test_mcdonald_schroder_put_call_symmetry(self):
        """C(S, K, r, q) = P(K, S, q, r) for American options, exactly on the tree."""
        for s, k, r, q in [(100.0, 90.0, 0.08, 0.05), (80.0, 100.0, 0.03, 0.06)]:
            call = opt(spot=s, strike=k, r=r, q=q, vol=0.3, kind=CALL)
            put = opt(spot=k, strike=s, r=q, q=r, vol=0.3, kind=PUT)
            assert binomial_price(call, 1000) == pytest.approx(binomial_price(put, 1000), rel=1e-10)


class TestValidation:
    @pytest.mark.parametrize("steps", [0, -5, 2.5, True, "100"])
    def test_bad_steps(self, steps):
        with pytest.raises(ValueError, match="steps must be a positive integer"):
            binomial_price(opt(), steps)

    def test_numpy_integer_steps_accepted(self):
        assert binomial_price(opt(), np.int64(50)) == binomial_price(opt(), 50)

    def test_coarse_tree_with_large_carry_rejected(self):
        p = opt(t=10.0, r=0.5, vol=0.01, kind=CALL)
        with pytest.raises(ValueError, match="outside \\(0, 1\\)"):
            binomial_price(p, 1)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be 'crr' or 'baw'"):
            american_price(opt(), method="lsm")  # type: ignore[arg-type]

    def test_returns_native_floats(self):
        for p in (opt(), opt(kind=CALL, q=0.05), opt(spot=40.0)):
            assert type(binomial_price(p, 50)) is float
            assert type(barone_adesi_whaley_price(p)) is float

    def test_one_step_tree_by_hand(self):
        """A single period, checked against the textbook formula."""
        p = opt(spot=100.0, strike=100.0, t=1.0, r=0.05, vol=0.2)
        u, d = math.exp(0.2), math.exp(-0.2)
        prob = (math.exp(0.05) - d) / (u - d)
        euro = math.exp(-0.05) * (1 - prob) * (100.0 - 100.0 * d)
        assert binomial_price(p, 1, american=False) == pytest.approx(euro, rel=1e-14)
        assert binomial_price(p, 1) == pytest.approx(max(euro, 0.0), rel=1e-14)
