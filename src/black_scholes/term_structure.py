"""
Multi-expiry volatility term structure built from SVI slices.

A single :class:`~black_scholes.surface.SVICurve` describes the smile at one
expiry. :class:`TermStructure` stacks several of them, one per expiry, checks
that no pair of adjacent expiries admits calendar-spread arbitrage, and
interpolates total implied variance between expiries so an implied volatility
can be read off at any ``(k, T)`` inside the quoted range.

Calendar arbitrage
------------------
With ``k = ln(K/F_T)`` the log-moneyness against the forward of each expiry,
the surface is free of calendar-spread arbitrage exactly when total implied
variance ``w(k, T) = sigma_imp(k, T)^2 * T`` is non-decreasing in ``T`` at every
fixed ``k`` (Gatheral and Jacquier, arXiv:1204.0646, Lemma 2.1). For SVI slices
that is a property of each pair of *adjacent* expiries: if
``w_1 <= w_2 <= ... <= w_n`` pointwise, every other pair follows by
transitivity. Each adjacent pair is checked with two conditions:

1. the slices do not cross, using the existing
   :meth:`~black_scholes.surface.SVICurve.calendar_arbitrage_free`; and
2. the longer-dated slice lies *above* the shorter-dated one.

The second condition is needed because the crossing test is symmetric: a long
slice lying uniformly *below* the short one never crosses it, yet total
variance then falls with maturity everywhere, which is calendar arbitrage too.

Both conditions are numerical over a stated log-moneyness range
``[k_low, k_high]`` (default ``[-4, 4]``, the same as the pairwise check), and
say nothing outside it.

Interpolation
-------------
Between two adjacent expiries ``T_i < T < T_{i+1}`` total variance is linear in
time at fixed ``k``::

    w(k, T) = w_i(k) + (T - T_i) / (T_{i+1} - T_i) * (w_{i+1}(k) - w_i(k))

    sigma_imp(k, T) = sqrt(w(k, T) / T)

At a node it reproduces the slice exactly. Because it is a convex combination
of ``w_i`` and ``w_{i+1}``, it is non-decreasing in ``T`` whenever the nodes
are, so an arbitrage-free set of slices gives an interpolated surface that is
free of calendar arbitrage at every intermediate maturity too. Linear
interpolation in total variance is equivalent to a flat *forward* variance
between expiries, the usual convention for a variance term structure.

It does **not** in general preserve butterfly-arbitrage freedom of the
intermediate smiles: a convex combination of two density-positive SVI slices
need not be density-positive. Use :meth:`TermStructure.butterfly_arbitrage_free`
to check an intermediate maturity explicitly.

Extrapolation
-------------
Maturities outside ``[T_1, T_n]`` are **rejected** with ``ValueError``. Any
extrapolation rule (flat implied volatility before the first expiry,
flat forward variance after the last) is an assumption the quotes do not
support, and picking one silently would make an out-of-range number look as
trustworthy as an interpolated one. Callers who want one can apply it to
:meth:`TermStructure.total_variance` at the end points themselves.

>>> from black_scholes.surface import SVICurve
>>> ts = TermStructure.from_slices({
...     0.5: SVICurve(0.02, 0.10, -0.30, 0.0, 0.15),
...     1.0: SVICurve(0.04, 0.10, -0.30, 0.0, 0.15),
... })
>>> ts.expiries
(0.5, 1.0)
>>> round(ts.total_variance(0.0, 0.75), 10)
0.045
>>> round(ts.implied_volatility(0.0, 0.75), 10)
0.2449489743
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from black_scholes.surface import SVICurve, SVIError, VolSurface

__all__ = [
    "CalendarViolation",
    "TermStructure",
]

#: Grid used to decide which of two non-crossing slices lies above the other.
#: The crossing test has already established that the sign of the difference is
#: uniform over the range, so this only has to read that sign, not search for
#: a change in it.
_ORDER_GRID_POINTS = 513


@dataclass(frozen=True, slots=True)
class CalendarViolation:
    """One pair of adjacent expiries that admits calendar-spread arbitrage.

    Attributes:
        short_expiry: The earlier of the two expiries, in years.
        long_expiry: The later of the two expiries, in years.
        reason: ``"cross"`` if the two slices intersect, ``"inverted"`` if the
            later slice lies below the earlier one everywhere in the range.

    >>> v = CalendarViolation(0.5, 1.0, "inverted")
    >>> str(v)
    'expiries 0.5 and 1: total variance decreases with maturity'
    >>> str(CalendarViolation(0.5, 1.0, "cross"))
    'expiries 0.5 and 1: slices cross'
    """

    short_expiry: float
    long_expiry: float
    reason: Literal["cross", "inverted"]

    def __str__(self) -> str:
        detail = (
            "slices cross" if self.reason == "cross" else "total variance decreases with maturity"
        )
        return f"expiries {self.short_expiry:g} and {self.long_expiry:g}: {detail}"


def _pair_violation(
    short_expiry: float,
    short: SVICurve,
    long_expiry: float,
    long: SVICurve,
    k_low: float,
    k_high: float,
) -> CalendarViolation | None:
    """Calendar check for one adjacent pair, or ``None`` if it is clean."""
    if not short.calendar_arbitrage_free(long, k_low=k_low, k_high=k_high):
        return CalendarViolation(short_expiry, long_expiry, "cross")

    # The slices do not cross, so ``long - short`` keeps one sign over the
    # range (up to the crossing test's tolerance). Read which sign it is. A
    # tolerance scaled to the difference admits identical slices and rounding
    # noise, matching the crossing test's treatment of a tangential touch.
    k = np.linspace(k_low, k_high, _ORDER_GRID_POINTS)
    difference = long.total_variance_vector(k) - short.total_variance_vector(k)
    scale = float(np.max(np.abs(difference)))
    if float(np.min(difference)) < -1e-12 * scale:
        return CalendarViolation(short_expiry, long_expiry, "inverted")
    return None


def _validate_expiry(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SVIError(f"{name} must be a finite number, got {value}")
    if value <= 0:
        raise SVIError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class TermStructure:
    """SVI slices at several expiries, with interpolation in time between them.

    Build one with :meth:`from_slices` (a mapping or pairs of expiry and
    slice, in any order) or :meth:`from_surfaces` (fit each single-expiry
    :class:`~black_scholes.surface.VolSurface`). The constructor takes the
    expiries already sorted.

    Args:
        expiries: Strictly increasing expiries in years, all positive.
        curves: One SVI slice per expiry, in total-variance space against
            log-moneyness ``k = ln(K/F_T)``.
        k_low: Lower edge of the log-moneyness range used by the calendar check.
        k_high: Upper edge of that range.
        check_calendar: Reject the slices if any adjacent pair admits calendar
            arbitrage. Pass ``False`` only to inspect a rejected set with
            :meth:`calendar_violations`; interpolated values from such a
            structure can decrease in maturity.

    Raises:
        SVIError: If there are no slices, the lengths differ, an expiry is
            not finite and positive, the expiries are not strictly increasing,
            a slice is not an :class:`SVICurve`, the range is inverted, or
            (with ``check_calendar``) an adjacent pair admits calendar
            arbitrage.

    >>> short = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
    >>> TermStructure((0.5, 1.0), (short, SVICurve(0.05, 0.10, -0.70, 0.0, 0.15)))
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: calendar arbitrage: expiries 0.5 and 1: slices cross
    >>> TermStructure((0.5, 1.0), (short, SVICurve(0.02, 0.10, -0.30, 0.0, 0.15)))
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: calendar arbitrage: expiries 0.5 and 1: total variance decreases with maturity
    >>> TermStructure((1.0, 0.5), (short, short))
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: expiries must be strictly increasing, got 1.0 then 0.5
    """

    expiries: tuple[float, ...]
    curves: tuple[SVICurve, ...]
    k_low: float = -4.0
    k_high: float = 4.0
    check_calendar: bool = True

    def __post_init__(self) -> None:
        if len(self.expiries) == 0:
            raise SVIError("TermStructure needs at least one expiry, got 0")
        if len(self.expiries) != len(self.curves):
            raise SVIError(
                f"expiries and curves differ in length: {len(self.expiries)} vs {len(self.curves)}"
            )
        for expiry in self.expiries:
            _validate_expiry("expiries", expiry)
        for earlier, later in itertools.pairwise(self.expiries):
            if not later > earlier:
                raise SVIError(f"expiries must be strictly increasing, got {earlier} then {later}")
        for curve in self.curves:
            if not isinstance(curve, SVICurve):
                raise SVIError(f"curves must be SVICurve instances, got {type(curve).__name__}")
        if not (math.isfinite(self.k_low) and math.isfinite(self.k_high)):
            raise SVIError(f"k_low and k_high must be finite, got {self.k_low}, {self.k_high}")
        if self.k_high <= self.k_low:
            raise SVIError(f"k_high ({self.k_high}) must be greater than k_low ({self.k_low})")
        if self.check_calendar:
            violations = self.calendar_violations()
            if violations:
                raise SVIError(f"calendar arbitrage: {violations[0]}")

    @classmethod
    def from_slices(
        cls,
        slices: Mapping[float, SVICurve] | Sequence[tuple[float, SVICurve]],
        *,
        k_low: float = -4.0,
        k_high: float = 4.0,
        check_calendar: bool = True,
    ) -> TermStructure:
        """Build from ``{expiry: slice}`` or ``[(expiry, slice), ...]`` in any order.

        Raises:
            SVIError: As for the constructor, and if an expiry appears twice.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> s2 = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> TermStructure.from_slices([(2.0, s2), (1.0, s1)]).expiries
        (1.0, 2.0)
        >>> TermStructure.from_slices([(1.0, s1), (1.0, s2)])
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: duplicate expiry 1.0
        """
        pairs = list(slices.items()) if isinstance(slices, Mapping) else list(slices)
        checked = [(_validate_expiry("expiries", expiry), curve) for expiry, curve in pairs]
        checked.sort(key=lambda pair: pair[0])
        for (earlier, _), (later, _) in itertools.pairwise(checked):
            if earlier == later:
                raise SVIError(f"duplicate expiry {earlier}")
        return cls(
            expiries=tuple(expiry for expiry, _ in checked),
            curves=tuple(curve for _, curve in checked),
            k_low=k_low,
            k_high=k_high,
            check_calendar=check_calendar,
        )

    @classmethod
    def from_surfaces(
        cls,
        surfaces: Sequence[VolSurface],
        *,
        weights: Literal["uniform", "vega"] = "vega",
        check_butterfly: bool = True,
        max_vol_error: float | None = 0.10,
        k_low: float = -4.0,
        k_high: float = 4.0,
        check_calendar: bool = True,
    ) -> TermStructure:
        """Fit an SVI slice to each single-expiry smile and stack the results.

        Each surface is fitted with :meth:`VolSurface.fit` using the given
        options, against log-moneyness relative to its own forward.

        Raises:
            SVIError: If any fit fails (the expiry is named), two surfaces share
                an expiry, or the fitted slices admit calendar arbitrage.

        >>> from black_scholes.surface import VolSurface
        >>> strikes = (80.0, 90.0, 100.0, 110.0, 120.0)
        >>> near = VolSurface(100.0, strikes, (0.26, 0.23, 0.21, 0.205, 0.21), 0.5, 0.0, 0.0)
        >>> far = VolSurface(100.0, strikes, (0.25, 0.23, 0.215, 0.21, 0.212), 1.0, 0.0, 0.0)
        >>> ts = TermStructure.from_surfaces([far, near])
        >>> ts.expiries
        (0.5, 1.0)
        """
        pairs: list[tuple[float, SVICurve]] = []
        for surface in surfaces:
            try:
                fit = surface.fit(
                    check_butterfly=check_butterfly,
                    max_vol_error=max_vol_error,
                    weights=weights,
                )
            except SVIError as exc:
                raise SVIError(f"expiry {surface.time_to_maturity:g}: {exc}") from exc
            pairs.append((surface.time_to_maturity, fit.curve))
        return cls.from_slices(pairs, k_low=k_low, k_high=k_high, check_calendar=check_calendar)

    def calendar_violations(self) -> list[CalendarViolation]:
        """Every adjacent pair of expiries that admits calendar arbitrage.

        Empty when the structure is calendar-arbitrage free over
        ``[k_low, k_high]``. Checking adjacent pairs is sufficient: if total
        variance is non-decreasing from each expiry to the next, it is
        non-decreasing across any pair.

        >>> a = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> b = SVICurve(0.02, 0.10, -0.30, 0.0, 0.15)
        >>> ts = TermStructure((0.5, 1.0, 2.0), (a, b, a), check_calendar=False)
        >>> [str(v) for v in ts.calendar_violations()]
        ['expiries 0.5 and 1: total variance decreases with maturity']
        """
        violations: list[CalendarViolation] = []
        for index in range(len(self.expiries) - 1):
            violation = _pair_violation(
                self.expiries[index],
                self.curves[index],
                self.expiries[index + 1],
                self.curves[index + 1],
                self.k_low,
                self.k_high,
            )
            if violation is not None:
                violations.append(violation)
        return violations

    def calendar_arbitrage_free(self) -> bool:
        """Whether no adjacent pair of expiries admits calendar arbitrage.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> TermStructure.from_slices({1.0: s1}).calendar_arbitrage_free()
        True
        """
        return not self.calendar_violations()

    @property
    def min_expiry(self) -> float:
        """Earliest expiry in the structure."""
        return self.expiries[0]

    @property
    def max_expiry(self) -> float:
        """Latest expiry in the structure."""
        return self.expiries[-1]

    def _check_maturity(self, maturity: float) -> float:
        maturity = float(maturity)
        if not math.isfinite(maturity):
            raise ValueError(f"maturity must be a finite number, got {maturity}")
        if not self.min_expiry <= maturity <= self.max_expiry:
            raise ValueError(
                f"maturity {maturity:g} is outside the quoted range "
                f"[{self.min_expiry:g}, {self.max_expiry:g}]; extrapolation in time "
                "is not supported"
            )
        return maturity

    def total_variance(self, k: float, maturity: float) -> float:
        """Total implied variance ``w(k, T)``, linear in ``T`` between expiries.

        ``w(k, T) = w_i(k) + (T - T_i)/(T_{i+1} - T_i) * (w_{i+1}(k) - w_i(k))``
        for ``T_i <= T <= T_{i+1}``; exactly ``w_i(k)`` at a node.

        Raises:
            ValueError: If ``k`` or ``maturity`` is not finite, or ``maturity``
                is outside ``[min_expiry, max_expiry]``.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> s2 = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> ts = TermStructure.from_slices({1.0: s1, 2.0: s2})
        >>> ts.total_variance(0.0, 1.0) == s1.total_variance(0.0)
        True
        >>> round(ts.total_variance(0.0, 1.5), 10)
        0.08
        >>> ts.total_variance(0.0, 2.5)
        Traceback (most recent call last):
            ...
        ValueError: maturity 2.5 is outside the quoted range [1, 2]; extrapolation in time is not supported
        """
        k = float(k)
        if not math.isfinite(k):
            raise ValueError(f"k must be a finite number, got {k}")
        maturity = self._check_maturity(maturity)

        index = bisect.bisect_left(self.expiries, maturity)
        if self.expiries[index] == maturity:
            return self.curves[index].total_variance(k)

        t_lo, t_hi = self.expiries[index - 1], self.expiries[index]
        w_lo = self.curves[index - 1].total_variance(k)
        w_hi = self.curves[index].total_variance(k)
        weight = (maturity - t_lo) / (t_hi - t_lo)
        return w_lo + weight * (w_hi - w_lo)

    def total_variance_vector(self, k: np.ndarray, maturity: float) -> np.ndarray:
        """Vectorised :meth:`total_variance` over ``k`` at one maturity.

        >>> import numpy as np
        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> s2 = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> ts = TermStructure.from_slices({1.0: s1, 2.0: s2})
        >>> np.round(ts.total_variance_vector(np.array([-0.1, 0.0, 0.1]), 1.5), 8)
        array([0.08602776, 0.08      , 0.08002776])
        """
        k_arr = np.asarray(k, dtype=float)
        if not np.all(np.isfinite(k_arr)):
            raise ValueError("k must be finite numbers")
        maturity = self._check_maturity(maturity)

        index = bisect.bisect_left(self.expiries, maturity)
        if self.expiries[index] == maturity:
            return self.curves[index].total_variance_vector(k_arr)

        t_lo, t_hi = self.expiries[index - 1], self.expiries[index]
        w_lo = self.curves[index - 1].total_variance_vector(k_arr)
        w_hi = self.curves[index].total_variance_vector(k_arr)
        weight = (maturity - t_lo) / (t_hi - t_lo)
        result: np.ndarray = w_lo + weight * (w_hi - w_lo)
        return result

    def implied_volatility(self, k: float, maturity: float) -> float:
        """Implied volatility ``sqrt(w(k, T) / T)`` at log-moneyness ``k``.

        Raises:
            ValueError: As for :meth:`total_variance`.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> s2 = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> ts = TermStructure.from_slices({1.0: s1, 2.0: s2})
        >>> ts.implied_volatility(0.0, 1.0) == s1.implied_volatility(0.0, 1.0)
        True
        >>> round(ts.implied_volatility(0.0, 1.5), 10)
        0.2309401077
        """
        variance = self.total_variance(k, maturity)
        return math.sqrt(variance / float(maturity))

    def implied_volatility_at_strike(self, strike: float, maturity: float, forward: float) -> float:
        """Implied volatility at an absolute strike, given that expiry's forward.

        Converts to ``k = ln(strike / forward)`` and calls
        :meth:`implied_volatility`. The forward must be the one for
        ``maturity``, typically ``S*exp((r - q)*T)``.

        Raises:
            ValueError: If ``strike`` or ``forward`` is not finite and positive,
                or as for :meth:`total_variance`.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> ts = TermStructure.from_slices({1.0: s1})
        >>> round(ts.implied_volatility_at_strike(100.0, 1.0, 100.0), 10)
        0.234520788
        >>> ts.implied_volatility_at_strike(-1.0, 1.0, 100.0)
        Traceback (most recent call last):
            ...
        ValueError: strike must be positive, got -1.0
        """
        for name, value in (("strike", strike), ("forward", forward)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        return self.implied_volatility(math.log(strike / forward), maturity)

    def butterfly_arbitrage_free(
        self, maturity: float, k_low: float = -2.0, k_high: float = 2.0, points: int = 401
    ) -> bool:
        """Gatheral's density condition on the interpolated smile at ``maturity``.

        Interpolating linearly in total variance keeps calendar-arbitrage
        freedom but not, in general, butterfly-arbitrage freedom, so an
        intermediate smile is worth checking before it is used to price. The
        condition is the one in
        :meth:`~black_scholes.surface.SVICurve.butterfly_arbitrage_free`, with
        ``w``, ``w'`` and ``w''`` taken from the interpolated slice (each is the
        same convex combination of the neighbouring slices' values).

        Raises:
            ValueError: If ``points`` is fewer than three, or as for
                :meth:`total_variance`.

        >>> s1 = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> s2 = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> TermStructure.from_slices({1.0: s1, 2.0: s2}).butterfly_arbitrage_free(1.5)
        True
        """
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")
        maturity = self._check_maturity(maturity)
        k = np.linspace(k_low, k_high, points)

        index = bisect.bisect_left(self.expiries, maturity)
        if self.expiries[index] == maturity:
            return self.curves[index].butterfly_arbitrage_free(k_low, k_high, points)

        t_lo, t_hi = self.expiries[index - 1], self.expiries[index]
        weight = (maturity - t_lo) / (t_hi - t_lo)

        def derivatives(curve: SVICurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            shifted = k - curve.m
            root = np.sqrt(shifted * shifted + curve.sigma * curve.sigma)
            w = curve.total_variance_vector(k)
            w1 = curve.b * (curve.rho + shifted / root)
            w2 = curve.b * curve.sigma**2 / root**3
            return w, w1, w2

        lo, hi = derivatives(self.curves[index - 1]), derivatives(self.curves[index])
        w, w1, w2 = (low + weight * (high - low) for low, high in zip(lo, hi, strict=True))
        if np.any(w <= 0):
            return False
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2
        return bool(np.all(g >= 0.0))
