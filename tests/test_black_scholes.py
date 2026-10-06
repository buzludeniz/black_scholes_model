"""
Tests for the Black-Scholes implementation.

Coverage areas: closed-form pricing against reference values, put-call
parity, Greeks checked against finite differences, implied-volatility
round-trips and arbitrage rejection, Monte Carlo convergence and variance
reduction, parameter validation, and CLI/GUI smoke coverage.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from black_scholes import (
    Greeks,
    OptionParams,
    OptionType,
    PricingResult,
    arbitrage_bounds,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    monte_carlo_price,
    price_option,
)

ATM_CALL = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
ATM_PUT = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)


@pytest.fixture
def atm_call_params() -> OptionParams:
    """At-the-money call, 1 year, 20% vol, 5% rate."""
    return ATM_CALL


@pytest.fixture
def atm_put_params() -> OptionParams:
    return ATM_PUT


@pytest.fixture
def otm_call_params() -> OptionParams:
    """Out-of-the-money call."""
    return OptionParams(100.0, 120.0, 0.5, 0.03, 0.25, OptionType.CALL)


@pytest.fixture
def itm_put_params() -> OptionParams:
    """In-the-money put."""
    return OptionParams(80.0, 100.0, 0.25, 0.04, 0.30, OptionType.PUT)


@pytest.fixture
def div_params() -> OptionParams:
    """Call paying a continuous dividend."""
    return OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL, 0.02)


class TestPricing:
    """Closed-form prices against known reference values."""

    def test_atm_call_known_value(self, atm_call_params):
        # Hull, Options Futures and Other Derivatives, ch. 15.
        assert 10.45 < black_scholes_price(atm_call_params) < 10.46

    def test_atm_put_known_value(self, atm_put_params):
        assert 5.57 < black_scholes_price(atm_put_params) < 5.58

    def test_returns_native_float(self, atm_call_params):
        assert type(black_scholes_price(atm_call_params)) is float
        assert type(price_option(atm_call_params).price) is float

    def test_deep_itm_call_approaches_intrinsic(self):
        params = OptionParams(200.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
        price = black_scholes_price(params)
        intrinsic = 200.0 - 100.0 * math.exp(-0.05)
        assert abs(price - intrinsic) < 0.01

    def test_deep_otm_call_approaches_zero(self):
        params = OptionParams(50.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
        assert black_scholes_price(params) < 0.01

    def test_short_dte_converges_to_intrinsic(self):
        for t in (1e-6, 1e-5, 1e-4):
            params = OptionParams(110.0, 100.0, t, 0.05, 0.20, OptionType.CALL)
            price = black_scholes_price(params)
            assert abs(price - 10.0) < 0.02

    def test_price_never_negative(self):
        for opt in (OptionType.CALL, OptionType.PUT):
            for spot in (10.0, 100.0, 500.0):
                params = OptionParams(spot, 100.0, 0.1, 0.02, 0.05, opt)
                assert black_scholes_price(params) >= 0.0

    def test_volatility_monotonic(self):
        prev_call = prev_put = -1.0
        for vol in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
            call = black_scholes_price(OptionParams(100.0, 100.0, 1.0, 0.05, vol, OptionType.CALL))
            put = black_scholes_price(OptionParams(100.0, 100.0, 1.0, 0.05, vol, OptionType.PUT))
            assert call > prev_call
            assert put > prev_put
            prev_call, prev_put = call, put

    def test_moneyness_ordering(self):
        prices = [
            black_scholes_price(OptionParams(100.0, k, 1.0, 0.05, 0.2, OptionType.CALL))
            for k in (80.0, 90.0, 100.0, 110.0, 120.0)
        ]
        assert prices == sorted(prices, reverse=True)

    def test_longer_maturity_is_worth_more(self):
        prices = [
            black_scholes_price(OptionParams(100.0, 100.0, t, 0.05, 0.2, OptionType.CALL))
            for t in (0.25, 0.5, 1.0, 2.0, 5.0)
        ]
        assert prices == sorted(prices)

    def test_volatility_upper_limit(self):
        for vol in (50.0, 100.0, 200.0):
            call = black_scholes_price(OptionParams(100.0, 100.0, 1.0, 0.05, vol, OptionType.CALL))
            put = black_scholes_price(OptionParams(100.0, 100.0, 1.0, 0.05, vol, OptionType.PUT))
            assert abs(call - 100.0) < 0.1
            assert abs(put - 100.0 * math.exp(-0.05)) < 0.1


class TestPutCallParity:
    """``C - P == S*e^(-qT) - K*e^(-rT)`` must hold exactly."""

    @staticmethod
    def parity(S, K, T, r, q):
        return S * math.exp(-q * T) - K * math.exp(-r * T)

    def test_atm(self):
        diff = black_scholes_price(ATM_CALL) - black_scholes_price(ATM_PUT)
        assert abs(diff - self.parity(100.0, 100.0, 1.0, 0.05, 0.0)) < 1e-10

    def test_with_dividends(self, div_params):
        put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT, 0.02)
        diff = black_scholes_price(div_params) - black_scholes_price(put)
        assert abs(diff - self.parity(100.0, 100.0, 1.0, 0.05, 0.02)) < 1e-10

    def test_across_strikes(self):
        for k in (80.0, 90.0, 100.0, 110.0, 120.0):
            call = OptionParams(100.0, k, 0.5, 0.04, 0.22, OptionType.CALL)
            put = OptionParams(100.0, k, 0.5, 0.04, 0.22, OptionType.PUT)
            diff = black_scholes_price(call) - black_scholes_price(put)
            assert abs(diff - self.parity(100.0, k, 0.5, 0.04, 0.0)) < 1e-10


class TestGreeks:
    """Analytic Greeks checked against finite differences."""

    @staticmethod
    def _bump(params: OptionParams, **changes) -> OptionParams:
        base = {
            "spot": params.spot,
            "strike": params.strike,
            "time_to_maturity": params.time_to_maturity,
            "risk_free_rate": params.risk_free_rate,
            "volatility": params.volatility,
            "option_type": params.option_type,
            "dividend_yield": params.dividend_yield,
        }
        base.update(changes)
        return OptionParams(**base)

    def test_delta_matches_finite_difference(self, atm_call_params):
        h = 1e-5
        greeks = black_scholes_greeks(atm_call_params)
        up = black_scholes_price(self._bump(atm_call_params, spot=atm_call_params.spot * (1 + h)))
        fd = (up - black_scholes_price(atm_call_params)) / (atm_call_params.spot * h)
        assert abs(greeks.delta - fd) < 1e-5

    def test_gamma_matches_finite_difference(self, atm_call_params):
        h = 1e-4
        greeks = black_scholes_greeks(atm_call_params)
        up = black_scholes_greeks(
            self._bump(atm_call_params, spot=atm_call_params.spot * (1 + h))
        ).delta
        down = black_scholes_greeks(
            self._bump(atm_call_params, spot=atm_call_params.spot * (1 - h))
        ).delta
        fd = (up - down) / (2 * atm_call_params.spot * h)
        assert abs(greeks.gamma - fd) < 1e-5

    def test_vega_matches_finite_difference(self, atm_call_params):
        h = 1e-5
        greeks = black_scholes_greeks(atm_call_params)
        bumped = self._bump(atm_call_params, volatility=atm_call_params.volatility * (1 + h))
        fd = (
            (black_scholes_price(bumped) - black_scholes_price(atm_call_params))
            / (atm_call_params.volatility * h)
            * 0.01
        )
        assert abs(greeks.vega - fd) < 1e-6

    def test_theta_matches_finite_difference(self, atm_call_params):
        h = 1e-6
        greeks = black_scholes_greeks(atm_call_params)
        shorter = self._bump(atm_call_params, time_to_maturity=atm_call_params.time_to_maturity - h)
        longer = self._bump(atm_call_params, time_to_maturity=atm_call_params.time_to_maturity + h)
        fd = (black_scholes_price(shorter) - black_scholes_price(longer)) / (2 * h) / 365
        assert abs(greeks.theta - fd) < 1e-5

    def test_rho_matches_finite_difference(self, atm_call_params):
        h = 1e-5
        greeks = black_scholes_greeks(atm_call_params)
        bumped = self._bump(atm_call_params, risk_free_rate=atm_call_params.risk_free_rate + h)
        fd = (black_scholes_price(bumped) - black_scholes_price(atm_call_params)) / h * 0.01
        assert abs(greeks.rho - fd) < 1e-5

    def test_greeks_are_native_floats(self, atm_call_params):
        greeks = black_scholes_greeks(atm_call_params)
        assert isinstance(greeks, Greeks)
        for name, value in greeks.as_dict().items():
            assert type(value) is float, name

    def test_signs(self):
        call = black_scholes_greeks(ATM_CALL)
        put = black_scholes_greeks(ATM_PUT)
        assert call.delta > 0 > put.delta
        assert call.gamma > 0
        assert put.gamma > 0
        assert call.vega > 0
        assert put.vega > 0
        assert call.rho > 0 > put.rho

    def test_delta_bounds(self):
        for k in (10.0, 80.0, 100.0, 120.0, 190.0):
            params = OptionParams(100.0, k, 1.0, 0.05, 0.2, OptionType.CALL)
            assert 0.0 <= black_scholes_greeks(params).delta <= 1.0

    def test_put_call_greek_relations(self):
        """Call delta minus put delta is the discounted dividend yield."""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        call = black_scholes_greeks(OptionParams(S, K, T, r, 0.2, OptionType.CALL, q))
        put = black_scholes_greeks(OptionParams(S, K, T, r, 0.2, OptionType.PUT, q))
        assert abs((call.delta - put.delta) - math.exp(-q * T)) < 1e-12
        # Gamma and Vega identical for calls and puts.
        assert abs(call.gamma - put.gamma) < 1e-12
        assert abs(call.vega - put.vega) < 1e-12


class TestArbitrageBounds:
    """Prices must land inside no-arbitrage bounds."""

    def test_bounds_call_and_put(self):
        assert arbitrage_bounds(ATM_CALL) == pytest.approx((4.877058, 100.0), abs=1e-6)
        assert arbitrage_bounds(ATM_PUT) == pytest.approx((0.0, 95.122942), abs=1e-6)

    @pytest.mark.parametrize("spot", [50.0, 100.0, 150.0])
    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_model_price_within_bounds(self, spot, opt):
        params = OptionParams(spot, 100.0, 0.5, 0.04, 0.3, opt, 0.01)
        lo, hi = arbitrage_bounds(params)
        assert lo <= black_scholes_price(params) <= hi


class TestImpliedVolatility:
    """Inversion accuracy and arbitrage rejection."""

    def test_roundtrip_call(self, atm_call_params):
        price = black_scholes_price(atm_call_params)
        assert implied_volatility(price, atm_call_params) == pytest.approx(0.20, abs=1e-6)

    def test_roundtrip_put(self, atm_put_params):
        price = black_scholes_price(atm_put_params)
        assert implied_volatility(price, atm_put_params) == pytest.approx(0.20, abs=1e-6)

    @pytest.mark.parametrize("k", [80.0, 90.0, 100.0, 110.0, 120.0])
    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_across_moneyness(self, k, opt):
        params = OptionParams(100.0, k, 0.5, 0.04, 0.25, opt)
        price = black_scholes_price(params)
        assert implied_volatility(price, params) == pytest.approx(0.25, abs=1e-6)

    def test_with_dividends(self, div_params):
        price = black_scholes_price(div_params)
        assert implied_volatility(price, div_params) == pytest.approx(0.20, abs=1e-6)

    def test_rejects_below_intrinsic(self):
        with pytest.raises(ValueError, match="outside no-arbitrage bounds"):
            implied_volatility(0.01, ATM_CALL)

    def test_rejects_above_upper_bound(self):
        with pytest.raises(ValueError, match="outside no-arbitrage bounds"):
            implied_volatility(500.0, ATM_CALL)

    def test_put_bounds_rejected(self):
        with pytest.raises(ValueError, match="outside no-arbitrage bounds"):
            implied_volatility(200.0, ATM_PUT)
        with pytest.raises(ValueError, match="outside no-arbitrage bounds"):
            implied_volatility(-1.0, ATM_PUT)

    def test_unreachable_price_reports_bracket(self):
        """A price above the vol_upper bracket must be reported, not guessed."""
        params = ATM_CALL
        with pytest.raises(ValueError, match="not reachable"):
            implied_volatility(black_scholes_price(params), params, vol_upper=0.05)

    def test_lower_bound_accepted(self):
        """A price on the lower bound corresponds to near-zero volatility."""
        lo, _ = arbitrage_bounds(ATM_CALL)
        iv = implied_volatility(lo, ATM_CALL, vol_lower=1e-9)
        assert iv < 0.01

    def test_upper_bound_needs_infinite_volatility(self):
        """The call upper bound is only reachable as vol tends to infinity."""
        _, hi = arbitrage_bounds(ATM_CALL)
        with pytest.raises(ValueError, match="not reachable"):
            implied_volatility(hi, ATM_CALL)

    def test_upper_bound_reachable_with_wider_bracket(self):
        _, hi = arbitrage_bounds(ATM_CALL)
        iv = implied_volatility(hi, ATM_CALL, vol_upper=1e4)
        assert iv > 100.0

    def test_volatility_smile_roundtrip(self):
        """Non-flat smile: each point must invert back to its own vol."""
        smile = [(90.0, 0.28), (100.0, 0.20), (110.0, 0.24)]
        for k, vol in smile:
            params = OptionParams(100.0, k, 0.75, 0.035, vol, OptionType.CALL)
            price = black_scholes_price(params)
            assert implied_volatility(price, params) == pytest.approx(vol, abs=1e-6)


class TestMonteCarlo:
    """Simulation agreement with the closed form."""

    def test_converges_to_analytical(self, atm_call_params):
        mc, se = monte_carlo_price(atm_call_params, n_paths=200_000, seed=42)
        assert abs(mc - black_scholes_price(atm_call_params)) < 3 * se

    def test_put_converges(self, atm_put_params):
        mc, se = monte_carlo_price(atm_put_params, n_paths=200_000, seed=123)
        assert abs(mc - black_scholes_price(atm_put_params)) < 3 * se

    def test_seed_is_reproducible(self, atm_call_params):
        a = monte_carlo_price(atm_call_params, n_paths=20_000, seed=7)
        b = monte_carlo_price(atm_call_params, n_paths=20_000, seed=7)
        assert a == b

    def test_different_seeds_differ(self, atm_call_params):
        a = monte_carlo_price(atm_call_params, n_paths=20_000, seed=1)
        b = monte_carlo_price(atm_call_params, n_paths=20_000, seed=2)
        assert a != b

    def test_antithetic_reduces_standard_error(self, atm_call_params):
        _, se_anti = monte_carlo_price(atm_call_params, n_paths=100_000, seed=3, antithetic=True)
        _, se_plain = monte_carlo_price(atm_call_params, n_paths=100_000, seed=3, antithetic=False)
        assert se_anti < se_plain

    def test_more_paths_reduce_error(self, atm_call_params):
        errs = [
            monte_carlo_price(atm_call_params, n_paths=n, seed=5)[1]
            for n in (10_000, 100_000, 1_000_000)
        ]
        assert errs[0] > errs[1] > errs[2]

    def test_error_scales_as_one_over_sqrt_n(self, atm_call_params):
        _, se_small = monte_carlo_price(atm_call_params, n_paths=10_000, seed=9)
        _, se_big = monte_carlo_price(atm_call_params, n_paths=160_000, seed=9)
        # 16x the paths should cut the standard error by about 4x.
        assert 3.5 < se_small / se_big < 4.5

    def test_multiple_steps_still_converge(self, atm_call_params):
        mc, se = monte_carlo_price(atm_call_params, n_paths=100_000, n_steps=12, seed=4)
        assert abs(mc - black_scholes_price(atm_call_params)) < 3 * se

    def test_returns_native_floats(self, atm_call_params):
        price, se = monte_carlo_price(atm_call_params, n_paths=1_000, seed=1)
        assert type(price) is float
        assert type(se) is float

    @pytest.mark.parametrize("n_paths", [0, -1])
    def test_rejects_bad_path_count(self, atm_call_params, n_paths):
        with pytest.raises(ValueError, match="n_paths must be positive"):
            monte_carlo_price(atm_call_params, n_paths=n_paths)

    def test_rejects_bad_step_count(self, atm_call_params):
        with pytest.raises(ValueError, match="n_steps must be positive"):
            monte_carlo_price(atm_call_params, n_paths=100, n_steps=0)

    def test_rejects_too_few_paths_for_antithetic(self, atm_call_params):
        with pytest.raises(ValueError, match="at least 4"):
            monte_carlo_price(atm_call_params, n_paths=2, antithetic=True)

    def test_odd_path_count_accepted(self, atm_call_params):
        price, se = monte_carlo_price(atm_call_params, n_paths=10_001, seed=6)
        assert np.isfinite(price)
        assert np.isfinite(se)


class TestInputValidation:
    """Parameter constraints."""

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("spot", -100.0, "spot must be positive"),
            ("spot", 0.0, "spot must be positive"),
            ("strike", 0.0, "strike must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("time_to_maturity", 0.0, "time_to_maturity must be positive"),
            ("time_to_maturity", -1.0, "time_to_maturity must be positive"),
            ("volatility", 0.0, "volatility must be positive"),
            ("volatility", -0.2, "volatility must be positive"),
            ("risk_free_rate", -0.01, "risk_free_rate must be non-negative"),
            ("dividend_yield", -0.01, "dividend_yield must be non-negative"),
        ],
    )
    def test_invalid_params_rejected(self, field, value, message):
        kwargs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_maturity": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "dividend_yield": 0.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            OptionParams(**kwargs)

    def test_non_option_type_rejected(self):
        with pytest.raises(TypeError, match="option_type must be OptionType"):
            OptionParams(100.0, 100.0, 1.0, 0.05, 0.2, option_type="call")

    def test_zero_rate_and_dividend_allowed(self):
        params = OptionParams(100.0, 100.0, 1.0, 0.0, 0.2, OptionType.CALL, 0.0)
        assert black_scholes_price(params) > 0

    def test_replace_volatility_returns_copy(self):
        original = ATM_CALL
        replaced = original.replace_volatility(0.35)
        assert replaced.volatility == 0.35
        assert original.volatility == 0.20

    def test_params_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ATM_CALL.spot = 200.0  # type: ignore[misc]

    def test_forward_price(self):
        assert ATM_CALL.forward_price == pytest.approx(100.0 * math.exp(0.05), abs=1e-10)


class TestPricingResult:
    """Result container behaviour."""

    def test_str_lists_all_greeks(self, atm_call_params):
        text = str(price_option(atm_call_params))
        for label in ("Price:", "Delta:", "Gamma:", "Vega:", "Theta:", "Rho:"):
            assert label in text

    def test_greeks_dict_keys(self, atm_call_params):
        result = price_option(atm_call_params)
        assert set(result.greeks.as_dict()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_result_carries_params(self, atm_call_params):
        result = price_option(atm_call_params)
        assert isinstance(result, PricingResult)
        assert result.params == atm_call_params

    def test_greeks_dict_is_json_serialisable(self, atm_call_params):
        payload = json.dumps(price_option(atm_call_params).greeks.as_dict())
        assert json.loads(payload)["delta"] == pytest.approx(0.636831, abs=1e-5)


class TestNumericalStability:
    """Edge cases that stress the floating point math."""

    def test_very_short_dte(self):
        params = OptionParams(100.0, 100.0, 1 / 365, 0.05, 0.20, OptionType.CALL)
        assert black_scholes_price(params) >= 0.0

    def test_very_low_volatility(self):
        params = OptionParams(100.0, 100.0, 1.0, 0.05, 0.001, OptionType.CALL)
        price = black_scholes_price(params)
        intrinsic = max(100.0 - 100.0 * math.exp(-0.05), 0.0)
        assert abs(price - intrinsic) < 0.01

    def test_high_rate(self):
        params = OptionParams(100.0, 100.0, 1.0, 0.50, 0.20, OptionType.CALL)
        base = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
        assert black_scholes_price(params) > black_scholes_price(base)

    def test_large_numbers(self):
        params = OptionParams(1e6, 1e6, 1.0, 0.05, 0.2, OptionType.CALL)
        price = black_scholes_price(params)
        assert np.isfinite(price)
        assert price == pytest.approx(1e6 * 0.10450584, rel=1e-4)

    def test_all_outputs_finite(self):
        params = OptionParams(123.45, 117.0, 0.37, 0.041, 0.26, OptionType.PUT, 0.008)
        result = price_option(params)
        assert np.isfinite(result.price)
        assert all(np.isfinite(v) for v in result.greeks.as_dict().values())
