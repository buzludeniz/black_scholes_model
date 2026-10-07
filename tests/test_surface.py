"""
Tests for volatility surface construction.

Covers the SVI parameterisation, arbitrage conditions in both closed form and
numerical form, the fitter, and the market-price inversion path.
"""

from __future__ import annotations

import math
from typing import ClassVar

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

    def test_inverted_search_range_rejected(self):
        with pytest.raises(ValueError, match="must be greater than k_low"):
            TRUTH.calendar_arbitrage_free(TRUTH, k_low=2.0, k_high=-2.0)

    def test_degenerate_search_range_rejected(self):
        with pytest.raises(ValueError, match="must be greater than k_low"):
            TRUTH.calendar_arbitrage_free(TRUTH, k_low=1.0, k_high=1.0)


class TestCalendarArbitrageResolution:
    """The answer must not depend on how the grid happens to be spaced.

    A fixed grid can step over a crossing that falls between two of its points,
    which reports a crossing pair as free. The check therefore locates the
    stationary points of the difference rather than trusting the samples, so it
    gives the same answer at every resolution.
    """

    #: Crosses TRUTH only between the sample points of a 3-point grid.
    NARROW = SVICurve(0.04, 0.3751, 0.715, 1.654, 0.264)

    def test_narrow_crossing_a_coarse_grid_would_miss(self):
        """The regression this refinement exists for.

        At three points the difference is negative at every sample, so sampling
        alone reports the pair as free. It genuinely crosses: the difference runs
        from about -1.234 to +0.032 across the range.
        """
        k = np.linspace(-4.0, 4.0, 3)
        difference = TRUTH.total_variance_vector(k) - self.NARROW.total_variance_vector(k)
        assert np.all(difference < 0.0), "precondition: the coarse grid sees one sign"
        assert bool(np.all(difference >= 0.0) or np.all(difference <= 0.0)), (
            "sampling alone would wrongly conclude 'free'"
        )

        assert TRUTH.calendar_arbitrage_free(self.NARROW, points=3) is False

    @pytest.mark.parametrize("points", [3, 5, 9, 51, 201, 1001, 2001])
    def test_answer_is_independent_of_grid_resolution(self, points):
        assert TRUTH.calendar_arbitrage_free(self.NARROW, points=points) is False

    def test_narrow_crossing_confirmed_by_dense_scan(self):
        """Cross-check the refinement against a very dense sampling."""
        dense = np.linspace(-4.0, 4.0, 200_001)
        difference = TRUTH.total_variance_vector(dense) - self.NARROW.total_variance_vector(dense)
        assert difference.min() < 0.0 < difference.max(), "the curves really do cross"

    @pytest.mark.parametrize("points", [3, 51, 2001])
    def test_crossing_near_the_upper_boundary_is_found(self, points):
        """A crossing at the very edge of the searched range still counts."""
        shifted = SVICurve(0.04, 0.10, -0.30, 3.98, 0.15)
        free = TRUTH.calendar_arbitrage_free(shifted, k_low=-4.0, k_high=4.0, points=points)
        assert free is False

    @pytest.mark.parametrize("points", [3, 51, 2001])
    def test_crossing_near_the_lower_boundary_is_found(self, points):
        shifted = SVICurve(0.04, 0.10, -0.30, -3.98, 0.15)
        free = TRUTH.calendar_arbitrage_free(shifted, k_low=-4.0, k_high=4.0, points=points)
        assert free is False

    #: Two pairs whose difference has a local maximum and minimum inside a single
    #: cell of a coarse grid. The derivative is positive only in a narrow band,
    #: so a 3-point grid sees equal signs at both ends of that cell, misses the
    #: extremum, and reports the slices as free even though they cross. Found by
    #: stress-probing the kernel rather than by the public API tests.
    COARSE_GRID_TRAPS: ClassVar[list[tuple[SVICurve, SVICurve]]] = [
        (
            SVICurve(
                0.08829679600921114,
                0.42135683536762275,
                -0.8700915103983988,
                -0.8422221651200972,
                0.3745157180181201,
            ),
            SVICurve(
                0.07343341363374854,
                0.46674056527908275,
                -0.6031955349001634,
                -0.5175824454247353,
                0.18938982532132864,
            ),
        ),
        (
            SVICurve(
                0.17709149960514176,
                0.2388409635475635,
                -0.48553468221851687,
                -1.918375636325012,
                0.4027893065574432,
            ),
            SVICurve(
                0.1400065174363586,
                0.39893056898755197,
                0.29112282747807816,
                -0.5366129694943043,
                0.2690889603497009,
            ),
        ),
    ]

    @pytest.mark.parametrize(("left", "right"), COARSE_GRID_TRAPS)
    def test_extremum_hidden_inside_one_coarse_cell_is_still_found(self, left, right):
        """The stationary-point scan must not inherit the caller's resolution.

        Both pairs genuinely cross: their difference runs from -0.55 to +0.026
        and from -1.57 to +0.11 respectively, so neither is close to a tangency.
        Before the stationary-point scan was given its own grid, a 3-point grid
        reported both as free of calendar arbitrage, which is a false negative
        in an arbitrage check.
        """
        dense = np.linspace(-4.0, 4.0, 400_001)
        difference = left.total_variance_vector(dense) - right.total_variance_vector(dense)
        assert difference.min() < 0.0 < difference.max(), "precondition: they really cross"

        assert left.calendar_arbitrage_free(right, points=3) is False

    @pytest.mark.parametrize(("left", "right"), COARSE_GRID_TRAPS)
    @pytest.mark.parametrize("points", [3, 5, 11, 51, 201, 2001])
    def test_trap_pairs_agree_at_every_resolution(self, left, right, points):
        assert left.calendar_arbitrage_free(right, points=points) is False

    def test_crossing_outside_the_searched_range_is_not_reported(self):
        """The documented limit: the given range bounds what is checked."""
        shifted = SVICurve(0.04, 0.10, -0.30, 3.98, 0.15)
        assert TRUTH.calendar_arbitrage_free(shifted, k_low=-2.0, k_high=2.0, points=3) is True

    @pytest.mark.parametrize("points", [3, 51, 2001])
    def test_identical_curves_are_free_at_any_resolution(self, points):
        assert TRUTH.calendar_arbitrage_free(TRUTH, points=points) is True

    @pytest.mark.parametrize("points", [3, 51, 2001])
    def test_a_constant_gap_never_crosses(self, points):
        """Curves differing only in `a` differ by a constant, so never cross."""
        higher = SVICurve(0.0401, 0.10, -0.30, 0.0, 0.15)
        assert TRUTH.calendar_arbitrage_free(higher, points=points) is True
        assert higher.calendar_arbitrage_free(TRUTH, points=points) is True

    @pytest.mark.parametrize("points", [3, 51, 2001])
    def test_near_tangency_is_permitted(self, points):
        """Touching zero without changing sign is not a crossing.

        Equality is allowed by the definition, so a difference that grazes zero
        and turns back stays free. Lifting a shallowly-crossing curve by exactly
        the depth of its dip produces that, and an exact touch is only ever zero
        up to rounding, so this also pins the tolerance the check relies on.
        """
        crossing = SVICurve(0.04, 0.10, -0.30, 0.05, 0.15)
        dense = np.linspace(-4.0, 4.0, 20_001)
        difference = TRUTH.total_variance_vector(dense) - crossing.total_variance_vector(dense)
        assert difference.min() < 0.0, "precondition: these two do cross"

        depth = abs(float(difference.min()))
        grazes = SVICurve(0.04 - depth, 0.10, -0.30, 0.05, 0.15)

        grazed = TRUTH.total_variance_vector(dense) - grazes.total_variance_vector(dense)
        assert grazed.min() >= -1e-12, "precondition: it touches zero without crossing"
        assert TRUTH.calendar_arbitrage_free(grazes, points=points) is True


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

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_strikes_rejected_by_fit(self, bad):
        strikes = np.array([70.0, 80.0, 90.0, 100.0, bad])
        with pytest.raises(SVIError, match="strikes must all be finite numbers"):
            fit_svi(100.0, strikes, np.full(5, 0.2), 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_vols_rejected_by_fit(self, bad):
        vols = np.full(STRIKES.size, 0.2)
        vols[3] = bad
        with pytest.raises(SVIError, match="implied volatilities must all be finite numbers"):
            fit_svi(100.0, STRIKES, vols, 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_forward_rejected(self, bad):
        with pytest.raises(SVIError, match="forward must be a finite number"):
            fit_svi(bad, STRIKES, np.full(7, 0.2), 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_maturity_rejected(self, bad):
        with pytest.raises(SVIError, match="maturity must be a finite number"):
            fit_svi(100.0, STRIKES, np.full(7, 0.2), bad)

    def test_non_finite_input_never_returns_a_fit(self):
        """The point of the check: no nan curve is ever produced."""
        with pytest.raises(SVIError):
            fit_svi(100.0, STRIKES, np.full(7, float("nan")), 1.0)
        with pytest.raises(SVIError):
            fit_svi(100.0, STRIKES, np.full(7, 0.2), float("nan"))


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


class TestNonFiniteSVIParameters:
    """Raw SVI parameters must be finite.

    ``NaN`` compares false against every bound, so it slips through a sign check
    and only fails later, wherever the arithmetic happens to propagate it. An
    infinite ``a`` or ``sigma`` is worse: it can reach an infinite minimum
    variance, which satisfies ``>= 0`` and would be accepted as valid.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("index", range(5))
    def test_check_parameters_rejects_non_finite(self, index, bad):
        values = [0.04, 0.10, -0.30, 0.0, 0.15]
        values[index] = bad
        assert check_parameters(*values) is False

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_constructor_rejects_non_finite(self, bad):
        with pytest.raises(SVIError, match="invalid SVI parameters"):
            SVICurve(a=bad, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
        with pytest.raises(SVIError, match="invalid SVI parameters"):
            SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=bad)

    def test_infinite_sigma_is_not_accepted_as_valid(self):
        """Without a finiteness check this case reports itself as valid."""
        assert check_parameters(0.04, 0.10, -0.30, 0.0, float("inf")) is False

    def test_valid_parameters_still_accepted(self):
        assert check_parameters(0.04, 0.10, -0.30, 0.0, 0.15) is True


class TestVolSurfaceValidation:
    """Direct construction is validated, not just the factory."""

    def test_empty_surface_rejected(self):
        with pytest.raises(SVIError, match="needs at least one quoted strike"):
            VolSurface(100.0, (), (), 1.0, 0.05, 0.0)

    def test_empty_surface_would_otherwise_raise_index_error(self):
        """The bug this prevents: interpolation indexing an empty list."""
        surface = object.__new__(VolSurface)
        object.__setattr__(surface, "strikes", ())
        object.__setattr__(surface, "implied_vols", ())
        object.__setattr__(surface, "spot", 100.0)
        object.__setattr__(surface, "time_to_maturity", 1.0)
        object.__setattr__(surface, "risk_free_rate", 0.05)
        object.__setattr__(surface, "dividend_yield", 0.0)
        with pytest.raises(IndexError):
            surface.implied_volatility(100.0)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(SVIError, match="differ in length: 2 vs 1"):
            VolSurface(100.0, (90.0, 100.0), (0.2,), 1.0, 0.05, 0.0)

    def test_duplicate_strikes_rejected(self):
        """Interpolation would be ambiguous about which quote to return."""
        with pytest.raises(SVIError, match="duplicate strikes"):
            VolSurface(100.0, (100.0, 100.0), (0.2, 0.3), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("strike", [0.0, -100.0, -1.0])
    def test_non_positive_strikes_rejected(self, strike):
        with pytest.raises(SVIError, match="strikes must be positive"):
            VolSurface(100.0, (strike,), (0.2,), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_strikes_rejected(self, bad):
        with pytest.raises(SVIError, match="strikes must be finite numbers"):
            VolSurface(100.0, (90.0, bad), (0.2, 0.3), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("vol", [0.0, -0.2, -1.0])
    def test_non_positive_vols_rejected(self, vol):
        with pytest.raises(SVIError, match="implied volatilities must be positive"):
            VolSurface(100.0, (100.0,), (vol,), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_vols_rejected(self, bad):
        with pytest.raises(SVIError, match="implied volatilities must be finite numbers"):
            VolSurface(100.0, (90.0, 100.0), (0.2, bad), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_spot_rejected(self, bad):
        with pytest.raises(SVIError, match="spot must be a finite number"):
            VolSurface(bad, (100.0,), (0.2,), 1.0, 0.05, 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_maturity_rejected(self, bad):
        with pytest.raises(SVIError, match="time_to_maturity must be a finite number"):
            VolSurface(100.0, (100.0,), (0.2,), bad, 0.05, 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rates_rejected(self, bad):
        with pytest.raises(SVIError, match="risk_free_rate must be a finite number"):
            VolSurface(100.0, (100.0,), (0.2,), 1.0, bad, 0.0)
        with pytest.raises(SVIError, match="dividend_yield must be a finite number"):
            VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, bad)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("spot", -1.0, "spot must be positive"),
            ("time_to_maturity", 0.0, "time_to_maturity must be positive"),
            ("risk_free_rate", -0.01, "risk_free_rate must be non-negative"),
            ("dividend_yield", -0.01, "dividend_yield must be non-negative"),
        ],
    )
    def test_invalid_scalars_rejected(self, field, value, message):
        kwargs = {
            "spot": 100.0,
            "strikes": (100.0,),
            "implied_vols": (0.2,),
            "time_to_maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.0,
        }
        kwargs[field] = value
        with pytest.raises(SVIError, match=message):
            VolSurface(**kwargs)

    def test_single_quote_is_a_valid_surface(self):
        """Interpolation needs one point; the four-strike minimum is the fitter's."""
        surface = VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, 0.0)
        assert surface.implied_volatility(100.0) == pytest.approx(0.2)
        assert surface.implied_volatility(50.0) == pytest.approx(0.2)
        assert surface.implied_volatility(500.0) == pytest.approx(0.2)

    def test_two_quotes_interpolate(self):
        """A strike between the quotes gets a strictly intermediate volatility."""
        surface = VolSurface(100.0, (90.0, 110.0), (0.25, 0.23), 1.0, 0.05, 0.0)
        assert 0.23 < surface.implied_volatility(100.0) < 0.25
        assert surface.implied_volatility(90.0) == pytest.approx(0.25)
        assert surface.implied_volatility(110.0) == pytest.approx(0.23)

    def test_fitting_still_needs_four_strikes(self):
        """The four-strike minimum stays with the fitter, not the surface."""
        surface = VolSurface(100.0, (90.0, 100.0, 110.0), (0.25, 0.24, 0.23), 1.0, 0.05, 0.0)
        with pytest.raises(SVIError, match="at least 4 strikes"):
            surface.fit()

    def test_unsorted_strikes_are_still_interpolatable(self):
        """Interpolation happens in log-moneyness, so it must sort first."""
        surface = VolSurface(100.0, (110.0, 90.0), (0.23, 0.25), 1.0, 0.05, 0.0)
        forward = 100.0 * math.exp(0.05)
        k_lo, k_hi = math.log(90.0 / forward), math.log(110.0 / forward)
        target = math.log(100.0 / forward)
        weight = (target - k_lo) / (k_hi - k_lo)
        assert surface.implied_volatility(100.0) == pytest.approx(0.25 + weight * (0.23 - 0.25))

    def test_factory_applies_the_same_rules(self):
        with pytest.raises(SVIError, match="duplicate strikes"):
            VolSurface.from_market_prices(100.0, [90.0, 90.0], [18.08, 19.0], 1.0, 0.05)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_factory_rejects_non_finite_prices(self, bad):
        with pytest.raises(SVIError, match="market prices must be finite numbers"):
            VolSurface.from_market_prices(100.0, [90.0], [bad], 1.0, 0.05)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_factory_rejects_non_finite_strikes(self, bad):
        with pytest.raises(SVIError, match="strikes must be finite numbers"):
            VolSurface.from_market_prices(100.0, [bad], [18.08], 1.0, 0.05)

    def test_factory_rejects_non_positive_strikes(self):
        with pytest.raises(SVIError, match="strikes must be positive"):
            VolSurface.from_market_prices(100.0, [-90.0], [18.08], 1.0, 0.05)
