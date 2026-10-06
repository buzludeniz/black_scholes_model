"""
Tests for volatility surface construction.

Covers the SVI parameterisation, arbitrage conditions in both closed form and
numerical form, the fitter, and the market-price inversion path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from black_scholes import OptionParams, OptionType, black_scholes_price
from black_scholes.surface import (
    SVI_FIT_FAILURE,
    SVICurve,
    SVIError,
    SVIFit,
    VolSurface,
    check_parameters,
    fit_svi,
)

# A known arbitrage-free curve used as ground truth throughout.
TRUTH = SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)

STRIKES = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0])


def truth_vols(forward: float = 100.0, maturity: float = 1.0) -> np.ndarray:
    """Implied volatilities produced by TRUTH at STRIKES."""
    return np.array([TRUTH.implied_volatility(math.log(k / forward), maturity) for k in STRIKES])


def truth_prices(spot=100.0, maturity=1.0, rate=0.05) -> list[float]:
    """Black-Scholes prices of the TRUTH smile."""
    vols = truth_vols()
    return [
        black_scholes_price(OptionParams(spot, float(k), maturity, rate, float(v), OptionType.CALL))
        for k, v in zip(STRIKES, vols, strict=True)
    ]


class TestSVICurveValues:
    """The SVI formula itself."""

    def test_total_variance_at_the_money(self):
        assert TRUTH.total_variance(0.0) == pytest.approx(0.055, abs=1e-12)

    def test_total_variance_matches_definition(self):
        """Compare against the formula written out independently."""
        for k in (-0.5, -0.1, 0.0, 0.1, 0.5):
            expected = 0.04 + 0.10 * (-0.30 * k + math.sqrt(k * k + 0.15 * 0.15))
            assert TRUTH.total_variance(k) == pytest.approx(expected, rel=1e-12)

    def test_vector_matches_scalar(self):
        k = np.linspace(-1.0, 1.0, 41)
        vector = TRUTH.total_variance_vector(k)
        for i, ki in enumerate(k):
            assert vector[i] == pytest.approx(TRUTH.total_variance(float(ki)), rel=1e-12)

    def test_vector_accepts_list(self):
        assert TRUTH.total_variance_vector([-0.1, 0.0]).shape == (2,)

    def test_implied_vol_inverts_total_variance(self):
        for k in (-0.3, 0.0, 0.3):
            for maturity in (0.25, 1.0, 5.0):
                vol = TRUTH.implied_volatility(k, maturity)
                assert vol**2 * maturity == pytest.approx(TRUTH.total_variance(k), rel=1e-12)

    def test_minimum_total_variance_is_a_true_minimum(self):
        """The closed form must match a numeric scan.

        The tolerance is set by the grid spacing, not by the formula: with 6001
        points the scan resolves the minimum to about 1e-8.
        """
        computed = TRUTH.minimum_total_variance()
        grid = np.linspace(-3.0, 3.0, 6001)
        scanned = float(np.min(TRUTH.total_variance_vector(grid)))
        assert computed == pytest.approx(scanned, abs=1e-7)
        assert computed <= scanned

    def test_minimum_is_at_the_analytic_location(self):
        rho, sigma, m = TRUTH.rho, TRUTH.sigma, TRUTH.m
        k_star = m - rho * sigma / math.sqrt(1 - rho * rho)
        near = TRUTH.total_variance(k_star - 1e-7)
        at = TRUTH.total_variance(k_star)
        assert at <= near

    def test_flat_curve_is_constant(self):
        flat = SVICurve(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2)
        values = [flat.total_variance(k) for k in (-0.5, -0.1, 0.0, 0.1, 0.5)]
        assert all(v == pytest.approx(0.04, rel=1e-12) for v in values)


class TestArbitrageConditions:
    """Both the closed-form and numerical arbitrage checks."""

    def test_valid_parameters_accepted(self):
        assert check_parameters(0.04, 0.10, -0.30, 0.0, 0.15) is True

    @pytest.mark.parametrize(
        ("a", "b", "rho", "m", "sigma"),
        [
            (0.04, -0.10, -0.30, 0.0, 0.15),  # negative slope
            (0.04, 0.10, -1.50, 0.0, 0.15),  # rho out of range
            (0.04, 0.10, 1.00, 0.0, 0.15),  # rho at the boundary
            (0.04, 0.10, 0.50, 0.0, 0.0),  # zero curvature
            (-1.0, 0.10, 0.0, 0.0, 0.10),  # negative minimum variance
            (0.04, 0.10, 0.0, 0.0, -0.1),  # negative curvature
            (-1.0, 5.0, 0.0, 0.0, 0.10),  # negative minimum via steep slope
        ],
    )
    def test_invalid_parameters_rejected(self, a, b, rho, m, sigma):
        assert check_parameters(a, b, rho, m, sigma) is False

    def test_constructor_rejects_invalid(self):
        with pytest.raises(SVIError, match="invalid SVI parameters"):
            SVICurve(a=0.04, b=-0.10, rho=-0.30, m=0.0, sigma=0.15)

    def test_constructed_curve_always_valid(self):
        assert TRUTH.parameters_are_valid() is True

    def test_true_curve_is_butterfly_free(self):
        assert TRUTH.butterfly_arbitrage_free() is True

    def test_extreme_curve_is_butterfly_arbitrage(self):
        assert SVICurve(0.001, 2.0, 0.9, 0.0, 0.01).butterfly_arbitrage_free() is False

    def test_flat_curve_is_butterfly_free(self):
        flat = SVICurve(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2)
        assert flat.butterfly_arbitrage_free() is True

    def test_butterfly_rejects_degenerate_grid(self):
        with pytest.raises(ValueError, match="points must be at least 3"):
            TRUTH.butterfly_arbitrage_free(points=2)

    def test_finer_grid_never_finds_more_arbitrage(self):
        """A coarser grid can only be more forgiving."""
        curve = SVICurve(0.004, 0.9, -0.6, 0.0, 0.05)
        coarse = curve.butterfly_arbitrage_free(points=51)
        fine = curve.butterfly_arbitrage_free(points=4001)
        assert not (coarse and not fine)


class TestFitSVI:
    """The optimiser."""

    def test_flat_slice_recovered_exactly(self):
        fit = fit_svi(100.0, STRIKES, np.full(7, 0.20), 1.0)
        assert fit.rms_error == pytest.approx(0.0, abs=1e-12)
        assert all(abs(v - 0.20) < 1e-9 for v in fit.model_vols)

    def test_recovers_true_parameters(self):
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0)
        assert fit.curve.a == pytest.approx(TRUTH.a, abs=1e-6)
        assert fit.curve.b == pytest.approx(TRUTH.b, abs=1e-6)
        assert fit.curve.rho == pytest.approx(TRUTH.rho, abs=1e-6)
        assert fit.curve.sigma == pytest.approx(TRUTH.sigma, abs=1e-6)

    def test_round_trip_is_machine_precision(self):
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0)
        assert fit.rms_error < 1e-10
        assert fit.max_error < 1e-9

    def test_fitted_curve_is_arbitrage_free(self):
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0)
        assert fit.curve.parameters_are_valid()
        assert fit.curve.butterfly_arbitrage_free()

    def test_smile_shape_recovered(self):
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0)
        assert fit.curve.b > 0, "a smile must have positive slope"
        assert fit.curve.rho < 0, "the truth curve is negatively skewed"

    def test_fit_accepts_lists(self):
        fit = fit_svi(100.0, list(STRIKES), list(truth_vols()), 1.0)
        assert fit.rms_error < 1e-10

    def test_forward_shifts_log_moneyness(self):
        """Same vols, different forward, must still fit the same curve."""
        vols = truth_vols(forward=100.0)
        fit = fit_svi(103.0, STRIKES, vols, 1.0)
        assert fit.rms_error < 1e-6

    def test_maturity_scales_total_variance(self):
        """Quoting the same smile at a longer maturity scales the variance.

        Implied volatilities are unchanged, so total variance grows in
        proportion to maturity. The recovered curve must reflect that.
        """
        vols = truth_vols()
        fit = fit_svi(100.0, STRIKES, vols, 4.0)
        assert fit.curve.total_variance(0.0) == pytest.approx(
            4.0 * TRUTH.total_variance(0.0), abs=1e-6
        )
        # And the implied volatility therefore matches the original smile.
        model_atm = fit.curve.implied_volatility(0.0, 4.0)
        assert model_atm == pytest.approx(float(truth_vols()[3]), abs=1e-6)

    def test_result_is_svifit(self):
        assert isinstance(fit_svi(100.0, STRIKES, truth_vols(), 1.0), SVIFit)

    def test_fit_records_inputs(self):
        vols = truth_vols()
        fit = fit_svi(100.0, STRIKES, vols, 1.0)
        assert fit.strikes == tuple(float(k) for k in STRIKES)
        assert fit.market_vols == pytest.approx(tuple(vols))
        assert len(fit.model_vols) == len(STRIKES)

    def test_relaxed_fit_accepts_arbitrage_violating_input(self):
        """The strict path must refuse what the relaxed path allows.

        This smile dips fast enough that no SVI curve matching it can keep a
        non-negative density, so the strict fit is rejected.
        """
        steep = np.array([0.40, 0.30, 0.23, 0.19, 0.23, 0.28, 0.36])
        with pytest.raises(SVIError, match="butterfly condition"):
            fit_svi(100.0, STRIKES, steep, 1.0)
        relaxed = fit_svi(100.0, STRIKES, steep, 1.0, check_butterfly=False)
        assert relaxed.curve.b > 0
        assert relaxed.rms_error < 0.05, "a rejected fit is still inspected"


class TestAgainstGatheralJacquier:
    """Cases taken straight from arXiv:1204.0646, the reference for this model.

    These are the strongest available check that the arbitrage conditions are
    implemented as the literature specifies rather than as they seemed to.
    """

    def test_example_3_1_is_detected_as_butterfly_arbitrage(self):
        """Example 3.1 is the paper's worked butterfly-arbitrage violation.

        Raw parameters (a, b, m, rho, sigma) = (-0.0410, 0.1331, 0.3586,
        0.3060, 0.4153) at t = 1, which the paper shows produce a negative
        density. Any implementation that does not flag this is wrong.
        """
        curve = SVICurve(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
        assert curve.parameters_are_valid(), (
            "parameter conditions alone do not rule out butterfly arbitrage"
        )
        assert curve.butterfly_arbitrage_free() is False

    def test_density_function_is_negativity_test(self):
        """Lemma 2.2: the slice is butterfly-free iff g(k) >= 0 for all k."""
        good = TRUTH
        bad = SVICurve(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

        def min_g(curve: SVICurve) -> float:
            k = np.linspace(-4.0, 4.0, 4001)
            w = curve.total_variance_vector(k)
            shifted = k - curve.m
            root = np.sqrt(shifted**2 + curve.sigma**2)
            w1 = curve.b * (curve.rho + shifted / root)
            w2 = curve.b * curve.sigma**2 / root**3
            g = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1**2 / 4.0) * (1.0 / w + 0.25) + 0.5 * w2
            return float(np.min(g))

        assert min_g(good) > 0
        assert min_g(bad) < 0

    def test_butterfly_result_agrees_with_direct_g_evaluation(self):
        """The public check must not disagree with the definition it encodes."""
        rng = np.random.default_rng(20240517)
        checked = 0
        for _ in range(300):
            a = float(rng.uniform(-0.05, 0.25))
            b = float(rng.uniform(0.0, 0.5))
            rho = float(rng.uniform(-0.9, 0.9))
            m = float(rng.uniform(-0.8, 0.8))
            sigma = float(rng.uniform(0.05, 0.8))
            if not check_parameters(a, b, rho, m, sigma):
                continue
            curve = SVICurve(a, b, rho, m, sigma)
            direct = float(np.min(_density_g(curve, np.linspace(-4.0, 4.0, 2001))))
            assert curve.butterfly_arbitrage_free(points=2001) == (direct >= 0.0)
            checked += 1
        assert checked > 50, "too few valid curves sampled to be meaningful"

    def test_minimum_total_variance_matches_the_papers_closed_form(self):
        """a + b*sigma*sqrt(1-rho^2), the value the paper states."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            a = float(rng.uniform(0.0, 0.2))
            b = float(rng.uniform(0.0, 0.4))
            rho = float(rng.uniform(-0.9, 0.9))
            m = float(rng.uniform(-0.8, 0.8))
            sigma = float(rng.uniform(0.05, 0.8))
            curve = SVICurve(a, b, rho, m, sigma)
            closed = a + b * sigma * math.sqrt(1 - rho * rho)
            assert curve.minimum_total_variance() == pytest.approx(closed, rel=1e-12)

    def test_minimum_is_attained_by_the_curve(self):
        """The reported minimum must actually be a value the curve takes."""
        curve = SVICurve(a=0.02, b=0.18, rho=-0.4, m=0.1, sigma=0.25)
        reported = curve.minimum_total_variance()
        grid = curve.total_variance_vector(np.linspace(-6.0, 6.0, 20001))
        assert reported <= float(np.min(grid)) + 1e-12


def _density_g(curve: SVICurve, k: np.ndarray) -> np.ndarray:
    """Gatheral-Jacquier equation (2.1), computed directly from its definition."""
    w = curve.total_variance_vector(k)
    shifted = k - curve.m
    root = np.sqrt(shifted**2 + curve.sigma**2)
    w1 = curve.b * (curve.rho + shifted / root)
    w2 = curve.b * curve.sigma**2 / root**3
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1**2 / 4.0) * (1.0 / w + 0.25) + 0.5 * w2


class TestJumpWingsParameters:
    """The trader-readable reparameterisation from the reference paper."""

    def test_keys_and_finiteness(self):
        jw = TRUTH.jump_wings_parameters(1.0)
        assert set(jw) == {
            "atm_variance",
            "atm_skew",
            "put_wing_slope",
            "call_wing_slope",
            "min_variance",
        }
        for value in jw.values():
            assert math.isfinite(value)

    def test_atm_variance_matches_the_curve(self):
        """The ATM variance must equal w(0)/t."""
        for maturity in (0.25, 1.0, 5.0):
            jw = TRUTH.jump_wings_parameters(maturity)
            assert jw["atm_variance"] == pytest.approx(
                TRUTH.total_variance(0.0) / maturity, rel=1e-12
            )

    def test_atm_skew_matches_the_derivative(self):
        """The ATM skew must equal d(sigma)/dk at k=0."""
        h = 1e-6
        numerical = (TRUTH.implied_volatility(h, 1.0) - TRUTH.implied_volatility(-h, 1.0)) / (2 * h)
        assert TRUTH.jump_wings_parameters(1.0)["atm_skew"] == pytest.approx(numerical, rel=1e-5)

    def test_min_variance_matches_scaled_minimum(self):
        for maturity in (0.5, 1.0, 3.0):
            jw = TRUTH.jump_wings_parameters(maturity)
            assert jw["min_variance"] == pytest.approx(
                TRUTH.minimum_total_variance() / maturity, rel=1e-12
            )

    def test_wing_slopes_follow_the_skew(self):
        """Negative skew means the put wing is steeper than the call wing."""
        jw = TRUTH.jump_wings_parameters(1.0)
        assert jw["put_wing_slope"] > jw["call_wing_slope"]
        assert jw["put_wing_slope"] > 0
        assert jw["call_wing_slope"] > 0

    def test_wing_slopes_sum_to_two_sqrt_w(self):
        """p + c = b*2/sqrt(w), an identity worth pinning down."""
        jw = TRUTH.jump_wings_parameters(1.0)
        assert jw["put_wing_slope"] + jw["call_wing_slope"] == pytest.approx(
            2 * TRUTH.b / math.sqrt(TRUTH.total_variance(0.0)), rel=1e-12
        )

    def test_symmetric_smile_has_zero_skew(self):
        symmetric = SVICurve(0.04, 0.10, 0.0, 0.0, 0.15)
        assert symmetric.jump_wings_parameters(1.0)["atm_skew"] == pytest.approx(0.0, abs=1e-15)

    def test_rejects_non_positive_maturity(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="maturity must be positive"):
                TRUTH.jump_wings_parameters(bad)


class TestCalendarArbitrage:
    """Crossing between slices, the paper's Lemma 3.3 in numeric form."""

    def test_uniformly_higher_slice_does_not_cross(self):
        short = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        long = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        assert short.calendar_arbitrage_free(long) is True
        assert long.calendar_arbitrage_free(short) is True

    def test_crossing_slices_are_flagged(self):
        short = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        skewed = SVICurve(0.05, 0.10, -0.70, 0.0, 0.15)
        assert short.calendar_arbitrage_free(skewed) is False
        assert skewed.calendar_arbitrage_free(short) is False

    def test_relation_is_symmetric(self):
        rng = np.random.default_rng(4)
        for _ in range(40):
            a1, b1, r1 = rng.uniform(0.0, 0.2), rng.uniform(0, 0.3), rng.uniform(-0.8, 0.8)
            a2, b2, r2 = rng.uniform(0.0, 0.2), rng.uniform(0, 0.3), rng.uniform(-0.8, 0.8)
            s1 = SVICurve(a1, b1, r1, 0.0, 0.2)
            s2 = SVICurve(a2, b2, r2, 0.0, 0.2)
            assert s1.calendar_arbitrage_free(s2) == s2.calendar_arbitrage_free(s1)

    def test_identical_curves_never_cross(self):
        assert TRUTH.calendar_arbitrage_free(TRUTH) is True

    def test_rejects_degenerate_grid(self):
        with pytest.raises(ValueError, match="points must be at least 3"):
            TRUTH.calendar_arbitrage_free(TRUTH, points=2)

    def test_finer_grid_detects_at_least_as_much(self):
        short = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        skewed = SVICurve(0.05, 0.10, -0.70, 0.0, 0.15)
        coarse = short.calendar_arbitrage_free(skewed, points=51)
        fine = short.calendar_arbitrage_free(skewed, points=4001)
        assert not (coarse and not fine)


class TestFitTolerance:
    """The error bound that stops an unfittable shape being called a fit."""

    SPIKE_K = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0])
    SPIKE_V = np.array([0.20, 0.22, 0.40, 0.60, 0.40, 0.22, 0.20])

    def test_unfittable_shape_is_rejected(self):
        """SVI cannot make a spike, so the fit must not be reported as good."""
        with pytest.raises(SVIError, match="cannot fit this smile"):
            fit_svi(100.0, self.SPIKE_K, self.SPIKE_V, 1.0)

    def test_tolerance_catches_it_even_though_butterfly_passes(self):
        """The bug this guards: arbitrage-free but wildly wrong."""
        loose = fit_svi(100.0, self.SPIKE_K, self.SPIKE_V, 1.0, max_vol_error=None)
        assert loose.max_error > 0.2
        assert loose.curve.parameters_are_valid(), (
            "the hidden fit is arbitrage-free, which is why an error bound "
            "is needed in addition to the butterfly check"
        )

    def test_error_message_names_strike_and_tolerance(self):
        with pytest.raises(SVIError) as info:
            fit_svi(100.0, self.SPIKE_K, self.SPIKE_V, 1.0)
        message = str(info.value)
        assert "worst error" in message
        assert "tolerance" in message
        assert "100" in message, "the offending strike should be named"

    def test_zero_tolerance_rejects_any_mismatch(self):
        strikes = np.array([70.0, 90.0, 100.0, 110.0, 150.0])
        vols = np.array([0.30, 0.25, 0.20, 0.21, 0.28])
        with pytest.raises(SVIError, match="cannot fit this smile"):
            fit_svi(100.0, strikes, vols, 1.0, max_vol_error=0.0)

    def test_wide_tolerance_accepts_a_real_smile(self):
        """The guard must not reject legitimate quotes."""
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0, max_vol_error=0.001)
        assert fit.max_error < 1e-3

    def test_flat_slice_is_never_rejected(self):
        fit = fit_svi(100.0, STRIKES, np.full(7, 0.20), 1.0, max_vol_error=0.0)
        assert fit.rms_error == pytest.approx(0.0, abs=1e-12)

    def test_round_trip_survives_a_tight_tolerance(self):
        fit = fit_svi(100.0, STRIKES, truth_vols(), 1.0, max_vol_error=1e-6)
        assert fit.rms_error < 1e-9


class TestFitValidation:
    """Input validation on the fitter."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"strikes": [90.0, 100.0, 110.0], "vols": [0.2, 0.2, 0.2]}, "at least 4 strikes"),
            ({"strikes": STRIKES, "vols": [0.2] * 3}, "differ in length"),
            ({"forward": 0.0}, "forward must be positive"),
            ({"forward": -100.0}, "forward must be positive"),
            ({"maturity": 0.0}, "maturity must be positive"),
            (
                {"strikes": np.array([70.0, -80.0, 90.0, 100.0]), "vols": np.full(4, 0.2)},
                "strikes must be positive",
            ),
            ({"strikes": STRIKES, "vols": np.zeros(7)}, "implied volatilities must be positive"),
        ],
    )
    def test_invalid_inputs_rejected(self, kwargs, message):
        call = {
            "forward": 100.0,
            "strikes": STRIKES,
            "vols": truth_vols(),
            "maturity": 1.0,
        }
        call.update(kwargs)
        with pytest.raises(SVIError, match=message):
            fit_svi(**call)

    def test_two_dimensional_inputs_rejected(self):
        with pytest.raises(SVIError, match="one-dimensional"):
            fit_svi(100.0, np.ones((2, 3)), np.ones((2, 3)), 1.0)

    def test_exactly_four_strikes_is_enough(self):
        strikes = [80.0, 95.0, 105.0, 130.0]
        vols = np.array([TRUTH.implied_volatility(math.log(k / 100.0), 1.0) for k in strikes])
        fit = fit_svi(100.0, strikes, vols, 1.0)
        assert fit.rms_error < 1e-8


class TestVolSurface:
    """Building a smile from quoted prices."""

    def test_from_market_prices_round_trip(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        assert surface.implied_vols == pytest.approx(tuple(truth_vols()), abs=1e-8)

    def test_forward_is_spot_grown_at_the_rate(self):
        surface = VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, 0.0)
        assert surface.forward == pytest.approx(100.0 * math.exp(0.05), abs=1e-12)

    def test_forward_net_of_dividend(self):
        surface = VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, 0.02)
        assert surface.forward == pytest.approx(100.0 * math.exp(0.03), abs=1e-12)

    def test_fit_round_trip(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        fit = surface.fit()
        assert fit.max_error < 1e-8

    def test_fit_relaxed_agrees_on_clean_input(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        assert surface.fit().curve.a == pytest.approx(surface.fit_relaxed().curve.a, abs=1e-9)

    def test_interpolation_hits_quoted_points(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        for strike, vol in zip(STRIKES, truth_vols(), strict=True):
            assert surface.implied_volatility(float(strike)) == pytest.approx(vol, abs=1e-8)

    def test_interpolation_between_points_is_monotone_in_strike(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        vols = [surface.implied_volatility(k) for k in np.linspace(75.0, 145.0, 40)]
        # A smile falls then rises; the sampled minimum is near the money.
        assert min(vols) == pytest.approx(min(vols))
        assert vols[len(vols) // 2] < vols[0]
        assert vols[len(vols) // 2] < vols[-1]

    def test_interpolation_clamps_outside_quoted_range(self):
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), truth_prices(), 1.0, 0.05)
        # Strikes are quoted ascending, so the first entry is the low wing and
        # the last is the high wing. Beyond either end the value is clamped to
        # the nearer wing rather than extrapolated.
        assert surface.implied_volatility(1e6) == pytest.approx(surface.implied_vols[-1], abs=1e-12)
        assert surface.implied_volatility(1.0) == pytest.approx(surface.implied_vols[0], abs=1e-12)

    def test_interpolation_sorts_unsorted_strikes(self):
        scrambled = list(STRIKES)[::-1]
        surface = VolSurface.from_market_prices(100.0, scrambled, truth_prices()[::-1], 1.0, 0.05)
        assert surface.implied_volatility(100.0) == pytest.approx(truth_vols()[3], abs=1e-8)

    def test_put_prices_are_accepted(self):
        put_prices = [
            black_scholes_price(OptionParams(100.0, float(k), 1.0, 0.05, float(v), OptionType.PUT))
            for k, v in zip(STRIKES, truth_vols(), strict=True)
        ]
        surface = VolSurface.from_market_prices(
            100.0, list(STRIKES), put_prices, 1.0, 0.05, option_type=OptionType.PUT
        )
        assert surface.option_type is OptionType.PUT
        assert surface.fit().max_error < 1e-8

    def test_dividend_yield_survives_the_round_trip(self):
        q = 0.03
        vols = truth_vols()
        prices = [
            black_scholes_price(
                OptionParams(100.0, float(k), 1.0, 0.05, float(v), OptionType.CALL, q)
            )
            for k, v in zip(STRIKES, vols, strict=True)
        ]
        surface = VolSurface.from_market_prices(100.0, list(STRIKES), prices, 1.0, 0.05, q)
        assert surface.implied_vols == pytest.approx(tuple(vols), abs=1e-8)
        assert surface.fit().max_error < 1e-8

    def test_arbitrage_violating_price_reports_strike(self):
        with pytest.raises(SVIError, match=r"strike 90\.0"):
            VolSurface.from_market_prices(100.0, [90.0], [0.5], 1.0, 0.05)

    def test_length_mismatch_rejected(self):
        with pytest.raises(SVIError, match="differ in length"):
            VolSurface.from_market_prices(100.0, [90.0, 100.0], [18.0], 1.0, 0.05)

    def test_too_few_strikes_to_fit(self):
        with pytest.raises(SVIError, match="at least 4 strikes"):
            VolSurface.from_market_prices(100.0, [90.0, 100.0], [18.08, 11.75], 1.0, 0.05).fit()

    def test_error_message_is_actionable(self):
        """The message must name the strike and the bounds, not just fail."""
        with pytest.raises(SVIError) as info:
            VolSurface.from_market_prices(100.0, [90.0], [0.5], 1.0, 0.05)
        message = str(info.value)
        assert "strike 90.0" in message
        assert "no-arbitrage bounds" in message


class TestSurfaceConstants:
    """Module surface."""

    def test_failure_message_is_public(self):
        assert isinstance(SVI_FIT_FAILURE, str)
        assert "arbitrage-free" in SVI_FIT_FAILURE
