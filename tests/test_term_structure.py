"""
Tests for the multi-expiry SVI term structure.

Covers construction and validation, the calendar-arbitrage check across every
adjacent pair of expiries (both the crossing and the inverted case), linear
interpolation of total variance in time, and the rejection of maturities
outside the quoted range.

There is no published numerical table for interpolating SVI slices in time, so
the reference checks here are analytic: a flat-volatility term structure must
return its volatility exactly at every intermediate maturity, linear
interpolation must reproduce the hand-computed convex combination, and the
no-calendar-arbitrage property of Gatheral and Jacquier (arXiv:1204.0646,
Lemma 2.1: total variance non-decreasing in T at fixed k) must hold on a dense
grid and translate into non-negative calendar-spread prices.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from black_scholes import OptionParams, OptionType, black_scholes_price
from black_scholes.surface import SVICurve, SVIError, VolSurface
from black_scholes.term_structure import CalendarViolation, TermStructure

# An arbitrage-free three-expiry fixture: the same smile shape with the level
# raised at each expiry, so each slice lies strictly above the one before.
SHORT = SVICurve(a=0.01, b=0.08, rho=-0.40, m=0.0, sigma=0.12)
MID = SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
LONG = SVICurve(a=0.09, b=0.12, rho=-0.25, m=0.05, sigma=0.20)
CLEAN = {0.25: SHORT, 1.0: MID, 2.0: LONG}

# Slices that cross MID: richer in the left wing, cheaper near the money.
CROSSING = SVICurve(a=0.05, b=0.10, rho=-0.70, m=0.0, sigma=0.15)
# A slice uniformly below MID. It never crosses MID, so the pairwise crossing
# test alone passes it, but placed later than MID total variance would fall.
BELOW_MID = SVICurve(a=0.02, b=0.10, rho=-0.30, m=0.0, sigma=0.15)


def flat(vol: float, maturity: float) -> SVICurve:
    """A flat smile with implied volatility ``vol`` at ``maturity``.

    With ``b = 0`` the SVI total variance is the constant ``a``.
    """
    return SVICurve(a=vol * vol * maturity, b=0.0, rho=0.0, m=0.0, sigma=0.1)


class TestConstruction:
    def test_from_mapping_sorts_expiries(self):
        ts = TermStructure.from_slices({2.0: LONG, 0.25: SHORT, 1.0: MID})
        assert ts.expiries == (0.25, 1.0, 2.0)
        assert ts.curves == (SHORT, MID, LONG)

    def test_from_pairs(self):
        ts = TermStructure.from_slices([(1.0, MID), (0.25, SHORT)])
        assert ts.expiries == (0.25, 1.0)
        assert ts.min_expiry == 0.25
        assert ts.max_expiry == 1.0

    def test_single_slice_is_valid(self):
        ts = TermStructure.from_slices({1.0: MID})
        assert ts.calendar_arbitrage_free()
        assert ts.implied_volatility(0.1, 1.0) == MID.implied_volatility(0.1, 1.0)

    def test_empty_rejected(self):
        with pytest.raises(SVIError, match="at least one expiry"):
            TermStructure.from_slices({})

    def test_length_mismatch_rejected(self):
        with pytest.raises(SVIError, match="differ in length: 2 vs 1"):
            TermStructure((0.5, 1.0), (MID,))

    def test_duplicate_expiry_rejected(self):
        with pytest.raises(SVIError, match=r"duplicate expiry 1\.0"):
            TermStructure.from_slices([(1.0, MID), (1.0, LONG)])

    def test_unsorted_constructor_rejected(self):
        with pytest.raises(SVIError, match="strictly increasing"):
            TermStructure((1.0, 0.25), (MID, SHORT))

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_bad_expiry_rejected(self, bad):
        with pytest.raises(SVIError, match="expiries must be"):
            TermStructure.from_slices({bad: MID})

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
    def test_bad_expiry_rejected_by_constructor(self, bad):
        with pytest.raises(SVIError, match="expiries must be"):
            TermStructure((bad,), (MID,))

    def test_non_curve_rejected(self):
        with pytest.raises(SVIError, match="SVICurve instances, got tuple"):
            TermStructure((1.0,), ((0.04, 0.1, -0.3, 0.0, 0.15),))  # type: ignore[arg-type]

    def test_inverted_range_rejected(self):
        with pytest.raises(SVIError, match="must be greater than"):
            TermStructure.from_slices(CLEAN, k_low=1.0, k_high=-1.0)

    def test_non_finite_range_rejected(self):
        with pytest.raises(SVIError, match="must be finite"):
            TermStructure.from_slices(CLEAN, k_high=math.inf)


class TestCalendarArbitrage:
    def test_clean_fixture_passes(self):
        ts = TermStructure.from_slices(CLEAN)
        assert ts.calendar_arbitrage_free()
        assert ts.calendar_violations() == []

    def test_clean_fixture_agrees_with_pairwise_check(self):
        """Every adjacent pair is also clean under the existing pairwise test."""
        for short, long in ((SHORT, MID), (MID, LONG)):
            assert short.calendar_arbitrage_free(long)

    def test_crossing_pair_rejected(self):
        with pytest.raises(SVIError, match=r"expiries 1 and 2: slices cross"):
            TermStructure.from_slices({0.25: SHORT, 1.0: MID, 2.0: CROSSING})

    def test_inverted_pair_rejected(self):
        """A later slice uniformly below an earlier one is calendar arbitrage.

        This pair never crosses, so the symmetric crossing check alone would
        pass it. The structure must still reject it.
        """
        assert MID.calendar_arbitrage_free(BELOW_MID)  # no crossing
        with pytest.raises(SVIError, match="total variance decreases with maturity"):
            TermStructure.from_slices({1.0: MID, 2.0: BELOW_MID})

    def test_every_adjacent_pair_is_checked(self):
        """A violation in the last pair is found as well as in the first."""
        ts = TermStructure.from_slices(
            {0.5: MID, 1.0: BELOW_MID, 1.5: MID, 2.0: CROSSING}, check_calendar=False
        )
        violations = ts.calendar_violations()
        assert violations == [
            CalendarViolation(0.5, 1.0, "inverted"),
            CalendarViolation(1.5, 2.0, "cross"),
        ]
        assert not ts.calendar_arbitrage_free()

    def test_only_adjacent_pairs_matter(self):
        """A clean chain implies every non-adjacent pair is ordered too."""
        ts = TermStructure.from_slices(CLEAN)
        k = np.linspace(ts.k_low, ts.k_high, 2001)
        assert np.all(LONG.total_variance_vector(k) >= SHORT.total_variance_vector(k))
        assert ts.calendar_arbitrage_free()

    def test_identical_slices_allowed(self):
        """Zero forward variance between two expiries is not arbitrage."""
        ts = TermStructure.from_slices({1.0: MID, 2.0: MID})
        assert ts.calendar_arbitrage_free()

    def test_crossing_outside_range_not_reported(self):
        """The check is a statement about [k_low, k_high], as documented."""
        edge = SVICurve(0.04, 0.10, -0.30, 3.98, 0.15)
        base = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        assert not base.calendar_arbitrage_free(edge)
        with pytest.raises(SVIError, match="calendar arbitrage"):
            TermStructure.from_slices({1.0: base, 2.0: edge})
        narrow = TermStructure.from_slices({1.0: base, 2.0: edge}, k_low=-2.0, k_high=2.0)
        assert narrow.calendar_arbitrage_free()

    def test_unchecked_structure_can_be_built_and_inspected(self):
        ts = TermStructure.from_slices({1.0: MID, 2.0: CROSSING}, check_calendar=False)
        assert ts.check_calendar is False
        assert [str(v) for v in ts.calendar_violations()] == ["expiries 1 and 2: slices cross"]


class TestInterpolation:
    def test_matches_slices_at_nodes(self):
        ts = TermStructure.from_slices(CLEAN)
        k = np.linspace(-1.5, 1.5, 61)
        for expiry, curve in CLEAN.items():
            for x in k:
                assert ts.total_variance(float(x), expiry) == curve.total_variance(float(x))
                assert ts.implied_volatility(float(x), expiry) == curve.implied_volatility(
                    float(x), expiry
                )
            np.testing.assert_array_equal(
                ts.total_variance_vector(k, expiry), curve.total_variance_vector(k)
            )

    def test_linear_in_total_variance(self):
        """Hand-computed convex combination at a quarter of the way."""
        ts = TermStructure.from_slices(CLEAN)
        for x in (-0.7, -0.2, 0.0, 0.3, 0.9):
            # 0.4375 is a quarter of the way from 0.25 to 1.0.
            expected = 0.75 * SHORT.total_variance(x) + 0.25 * MID.total_variance(x)
            assert ts.total_variance(x, 0.4375) == pytest.approx(expected, rel=1e-14)
            assert ts.implied_volatility(x, 0.4375) == pytest.approx(
                math.sqrt(expected / 0.4375), rel=1e-14
            )

    def test_vector_matches_scalar(self):
        ts = TermStructure.from_slices(CLEAN)
        k = np.linspace(-1.0, 1.0, 21)
        for maturity in (0.25, 0.6, 1.0, 1.7, 2.0):
            expected = [ts.total_variance(float(x), maturity) for x in k]
            np.testing.assert_allclose(ts.total_variance_vector(k, maturity), expected, rtol=1e-14)

    def test_flat_term_structure_reference(self):
        """Constant 20% volatility gives exactly 20% at every intermediate T.

        Analytic reference: with sigma constant, w(T) = sigma^2 T is linear, so
        linear interpolation in total variance must recover sigma exactly.
        """
        ts = TermStructure.from_slices({t: flat(0.20, t) for t in (0.1, 0.5, 1.0, 3.0)})
        for maturity in np.linspace(0.1, 3.0, 59):
            for k in (-1.0, 0.0, 0.5):
                assert ts.implied_volatility(k, float(maturity)) == pytest.approx(0.20, abs=1e-14)

    def test_flat_forward_variance_between_nodes(self):
        """Linear total variance means constant forward variance per segment.

        With 20% to T=1 and 30% to T=2, forward variance on [1, 2] is
        (0.09*2 - 0.04*1) / 1 = 0.14, so w(1.5) = 0.04 + 0.5*0.14 = 0.11.
        """
        ts = TermStructure.from_slices({1.0: flat(0.20, 1.0), 2.0: flat(0.30, 2.0)})
        assert ts.total_variance(0.0, 1.5) == pytest.approx(0.11, abs=1e-15)
        assert ts.implied_volatility(0.0, 1.5) == pytest.approx(math.sqrt(0.11 / 1.5), abs=1e-15)

    def test_total_variance_monotone_in_time(self):
        """w(k, T) is non-decreasing in T at every k across the whole range."""
        ts = TermStructure.from_slices(CLEAN)
        k = np.linspace(ts.k_low, ts.k_high, 401)
        maturities = np.linspace(ts.min_expiry, ts.max_expiry, 300)
        grid = np.array([ts.total_variance_vector(k, float(t)) for t in maturities])
        assert np.all(np.diff(grid, axis=0) >= -1e-15)

    def test_monotone_for_random_arbitrage_free_chains(self):
        """Stacking increasingly levelled slices always yields monotone w."""
        rng = np.random.default_rng(20261110)
        k = np.linspace(-3.0, 3.0, 121)
        for _ in range(25):
            b = rng.uniform(0.05, 0.2)
            rho = rng.uniform(-0.8, 0.2)
            sigma = rng.uniform(0.05, 0.3)
            levels = np.cumsum(rng.uniform(0.005, 0.05, size=4))
            expiries = np.cumsum(rng.uniform(0.1, 1.0, size=4))
            ts = TermStructure.from_slices(
                {
                    float(t): SVICurve(float(a), b, rho, 0.0, sigma)
                    for t, a in zip(expiries, levels, strict=True)
                }
            )
            maturities = np.linspace(ts.min_expiry, ts.max_expiry, 50)
            grid = np.array([ts.total_variance_vector(k, float(t)) for t in maturities])
            assert np.all(np.diff(grid, axis=0) >= -1e-15)

    def test_calendar_spreads_have_non_negative_value(self):
        """The economic meaning of monotone total variance.

        With zero rates and dividends, a call at fixed moneyness K/F is worth
        at least as much at a later maturity. Prices come from the
        interpolated implied volatilities through Black-Scholes.
        """
        ts = TermStructure.from_slices(CLEAN)
        for strike in (70.0, 90.0, 100.0, 115.0, 140.0):
            k = math.log(strike / 100.0)
            prices = [
                black_scholes_price(
                    OptionParams(100.0, strike, float(t), 0.0, ts.implied_volatility(k, float(t)))
                )
                for t in np.linspace(0.25, 2.0, 36)
            ]
            assert all(later >= earlier - 1e-12 for earlier, later in itertools.pairwise(prices))

    def test_implied_volatility_at_strike(self):
        ts = TermStructure.from_slices(CLEAN)
        forward = 105.0
        assert ts.implied_volatility_at_strike(90.0, 1.3, forward) == pytest.approx(
            ts.implied_volatility(math.log(90.0 / forward), 1.3), rel=1e-15
        )

    @pytest.mark.parametrize(("strike", "forward"), [(0.0, 100.0), (100.0, -1.0), (math.nan, 1.0)])
    def test_implied_volatility_at_strike_rejects_bad_inputs(self, strike, forward):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="must be positive"):
            ts.implied_volatility_at_strike(strike, 1.0, forward)


class TestOutOfRange:
    @pytest.mark.parametrize("maturity", [0.1, 0.2499, 2.0001, 5.0])
    def test_extrapolation_rejected(self, maturity):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="outside the quoted range"):
            ts.total_variance(0.0, maturity)
        with pytest.raises(ValueError, match="outside the quoted range"):
            ts.implied_volatility(0.0, maturity)
        with pytest.raises(ValueError, match="outside the quoted range"):
            ts.total_variance_vector(np.array([0.0]), maturity)

    def test_end_points_are_inside(self):
        ts = TermStructure.from_slices(CLEAN)
        assert ts.total_variance(0.0, 0.25) == SHORT.total_variance(0.0)
        assert ts.total_variance(0.0, 2.0) == LONG.total_variance(0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_maturity_rejected(self, bad):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="maturity must be a finite number"):
            ts.total_variance(0.0, bad)

    def test_non_finite_k_rejected(self):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="k must be a finite number"):
            ts.total_variance(math.nan, 1.0)
        with pytest.raises(ValueError, match="k must be finite"):
            ts.total_variance_vector(np.array([0.0, math.inf]), 1.0)


class TestButterflyAtIntermediateMaturity:
    def test_clean_structure_intermediate_smiles(self):
        ts = TermStructure.from_slices(CLEAN)
        for maturity in (0.25, 0.5, 1.0, 1.5, 2.0):
            assert ts.butterfly_arbitrage_free(maturity)

    def test_node_delegates_to_slice(self):
        bad = SVICurve(0.001, 2.0, 0.9, 0.0, 0.01)
        assert not bad.butterfly_arbitrage_free()
        ts = TermStructure.from_slices({1.0: bad})
        assert not ts.butterfly_arbitrage_free(1.0)

    def test_interpolated_smile_inherits_violation(self):
        """Halfway to a density-violating slice the smile still violates."""
        steep = SVICurve(0.001, 2.0, 0.9, 0.0, 0.01)
        base = SVICurve(0.0005, 0.5, 0.9, 0.0, 0.01)
        ts = TermStructure.from_slices({1.0: base, 2.0: steep}, check_calendar=False)
        assert not ts.butterfly_arbitrage_free(1.5)

    def test_matches_slice_formula_at_intermediate_flat_smile(self):
        ts = TermStructure.from_slices({1.0: flat(0.2, 1.0), 2.0: flat(0.25, 2.0)})
        assert ts.butterfly_arbitrage_free(1.5)

    def test_non_positive_variance_fails(self):
        zero = SVICurve(0.0, 0.0, 0.0, 0.0, 0.1)
        ts = TermStructure.from_slices({1.0: zero, 2.0: zero})
        assert not ts.butterfly_arbitrage_free(1.5)

    def test_points_validated(self):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="points must be at least 3"):
            ts.butterfly_arbitrage_free(1.0, points=2)

    def test_range_validated(self):
        ts = TermStructure.from_slices(CLEAN)
        with pytest.raises(ValueError, match="outside the quoted range"):
            ts.butterfly_arbitrage_free(3.0)


class TestFromSurfaces:
    STRIKES = (70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0)

    def surface_from(self, curve: SVICurve, maturity: float, rate: float = 0.03) -> VolSurface:
        spot = 100.0
        forward = spot * math.exp(rate * maturity)
        vols = tuple(
            curve.implied_volatility(math.log(k / forward), maturity) for k in self.STRIKES
        )
        return VolSurface(spot, self.STRIKES, vols, maturity, rate, 0.0)

    def test_recovers_slices_from_exact_smiles(self):
        surfaces = [self.surface_from(curve, t) for t, curve in CLEAN.items()]
        ts = TermStructure.from_surfaces(list(reversed(surfaces)))
        assert ts.expiries == (0.25, 1.0, 2.0)
        for expiry, curve in CLEAN.items():
            for k in (-0.3, 0.0, 0.3):
                assert ts.total_variance(k, expiry) == pytest.approx(
                    curve.total_variance(k), abs=1e-6
                )

    def test_from_market_prices_end_to_end(self):
        """Prices in, an interpolated volatility out."""
        surfaces = []
        for t, curve in CLEAN.items():
            exact = self.surface_from(curve, t)
            prices = [
                black_scholes_price(OptionParams(100.0, k, t, 0.03, v, OptionType.CALL))
                for k, v in zip(exact.strikes, exact.implied_vols, strict=True)
            ]
            surfaces.append(VolSurface.from_market_prices(100.0, self.STRIKES, prices, t, 0.03))
        ts = TermStructure.from_surfaces(surfaces)
        expected = 0.5 * (MID.total_variance(0.0) + LONG.total_variance(0.0))
        assert ts.total_variance(0.0, 1.5) == pytest.approx(expected, abs=1e-6)

    def test_fit_failure_names_expiry(self):
        too_few = VolSurface(100.0, (90.0, 100.0), (0.2, 0.2), 0.5, 0.0, 0.0)
        with pytest.raises(SVIError, match=r"^expiry 0.5: "):
            TermStructure.from_surfaces([too_few])

    def test_calendar_violation_in_fitted_slices(self):
        surfaces = [self.surface_from(MID, 1.0), self.surface_from(BELOW_MID, 2.0)]
        with pytest.raises(SVIError, match="calendar arbitrage"):
            TermStructure.from_surfaces(surfaces)
        ts = TermStructure.from_surfaces(surfaces, check_calendar=False)
        assert [v.reason for v in ts.calendar_violations()] == ["inverted"]

    def test_duplicate_expiries_rejected(self):
        surfaces = [self.surface_from(MID, 1.0), self.surface_from(MID, 1.0)]
        with pytest.raises(SVIError, match="duplicate expiry"):
            TermStructure.from_surfaces(surfaces)


class TestSharedDerivatives:
    """The butterfly checks reuse SVICurve's derivative helpers."""

    def test_vector_derivatives_match_scalar_and_finite_differences(self):
        k = np.linspace(-1.5, 1.5, 31)
        h = 1e-4
        for curve in (SHORT, MID, LONG):
            d1, d2 = curve._d1_vector(k), curve._d2_vector(k)
            for x, first, second in zip(k, d1, d2, strict=True):
                x = float(x)
                assert first == pytest.approx(curve._d1(x), rel=1e-14, abs=1e-15)
                assert second == pytest.approx(curve._d2(x), rel=1e-14)
                up, mid, down = (curve.total_variance(x + s) for s in (h, 0.0, -h))
                assert first == pytest.approx((up - down) / (2 * h), abs=1e-7)
                assert second == pytest.approx((up - 2 * mid + down) / (h * h), rel=1e-4)
