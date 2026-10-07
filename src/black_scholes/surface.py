"""
Volatility surface construction from market prices.

The Black-Scholes model assumes a single flat volatility, but real markets quote
a *smile*: implied volatility varies with strike. This module fits the SVI
parameterisation to a set of quoted prices and reports whether the result is
free of static arbitrage.

SVI (Stochastic Volatility Inspired) writes total implied variance as a function
of log-moneyness ``k = ln(K/F)``::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Five parameters, and unlike a polynomial fit the shape stays economically
sensible: a controllable smile, controlled skew, and wings that level off rather
than explode.

Static arbitrage has two faces here:

- **Parameter arbitrage** is checked in closed form: ``b >= 0``, ``|rho| < 1``,
  and a non-negative minimum total variance.
- **Butterfly arbitrage** is checked numerically with Gatheral's condition,
  because no closed form for it exists in the raw parameterisation. The paper
  notes that finding usable parameter conditions for it is "seemingly
  impossible", which is why a numerical check is the honest answer rather than a
  closed-form test.

A fit that cannot satisfy both is reported rather than quietly returned.

Reference
---------
J. Gatheral and A. Jacquier, *Arbitrage-free SVI volatility surfaces*,
arXiv:1204.0646. The raw parameterisation, the density condition used here, and
the closed-form minimum total variance are all taken from that paper, and its
Example 3.1 is used as a regression test.

>>> from black_scholes.surface import SVICurve
>>> curve = SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
>>> round(curve.total_variance(0.0), 10)
0.055
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from black_scholes import OptionParams, OptionType, implied_volatility

__all__ = [
    "SVI_FIT_FAILURE",
    "SVICurve",
    "SVIError",
    "SVIFit",
    "VolSurface",
    "check_parameters",
    "fit_svi",
]


class SVIError(ValueError):
    """Raised when a surface cannot be built or fails an arbitrage check."""


SVI_FIT_FAILURE = "SVI fit did not converge to an arbitrage-free parameter set"

#: Floor on the grid used to locate stationary points in the calendar check.
#:
#: The stationary-point scan must not inherit the caller's resolution, because a
#: coarse grid can hide a genuine local extremum inside a single cell. See
#: :meth:`SVICurve.calendar_arbitrage_free`. The derivative of a difference of
#: two SVI slices is smooth with a small number of zeros, so a few hundred
#: samples separate them reliably.
_CALENDAR_SEARCH_MIN_POINTS = 513


def check_parameters(a: float, b: float, rho: float, m: float, sigma: float) -> bool:
    """Whether raw SVI parameters satisfy the closed-form arbitrage conditions.

    Exposed separately from :meth:`SVICurve.parameters_are_valid` because the
    constructor rejects invalid parameters, so an instance can never report
    ``False``. The optimiser does produce invalid parameters, and this is what
    tests them.

    The conditions are that every parameter is finite, ``b >= 0``,
    ``|rho| < 1``, ``sigma > 0``, and a non-negative minimum total variance.

    Finiteness is checked explicitly. Left to the conditions below it would be
    accidental: ``NaN`` fails every comparison and so reaches the minimum
    variance, where ``NaN >= 0`` is false and it happens to be rejected, while
    an infinite ``a`` or ``sigma`` can propagate to an infinite minimum that
    compares ``>= 0`` as true and would be accepted.

    >>> check_parameters(0.04, 0.10, -0.30, 0.0, 0.15)
    True
    >>> check_parameters(0.04, -0.10, -0.30, 0.0, 0.15)
    False
    >>> check_parameters(0.04, 0.10, -1.50, 0.0, 0.15)
    False
    >>> check_parameters(0.04, 0.10, 0.50, 0.0, 0.0)
    False
    >>> check_parameters(-1.0, 0.10, 0.0, 0.0, 0.10)
    False
    >>> check_parameters(0.04, 0.10, 0.99, 0.0, 0.15)
    True
    >>> check_parameters(float("nan"), 0.10, -0.30, 0.0, 0.15)
    False
    >>> check_parameters(float("inf"), 0.10, -0.30, 0.0, 0.15)
    False
    >>> check_parameters(0.04, 0.10, -0.30, 0.0, float("inf"))
    False
    """
    if not all(math.isfinite(x) for x in (a, b, rho, m, sigma)):
        return False
    if b < 0:
        return False
    if not -1.0 < rho < 1.0:
        return False
    if sigma <= 0:
        return False
    # Minimum of w(k) over all real k, attained where the derivative vanishes.
    root = math.sqrt(1.0 - rho * rho)
    k_star = m - rho * sigma / root
    shifted = k_star - m
    minimum = a + b * (rho * shifted + math.sqrt(shifted * shifted + sigma * sigma))
    return minimum >= 0.0


@dataclass(frozen=True, slots=True)
class SVICurve:
    """A fitted SVI slice in total-variance space.

    Args:
        a: Level of total variance.
        b: Slope, controlling the smile. Must be non-negative.
        rho: Skew correlation, in ``(-1, 1)``.
        m: Horizontal shift of the smile, in log-moneyness.
        sigma: Wing curvature. Must be positive.

    Raises:
        SVIError: If the parameters violate an arbitrage condition.

    >>> curve = SVICurve(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
    >>> curve.total_variance(0.0)
    0.055
    >>> round(curve.total_variance(-0.20), 10)
    0.071
    >>> SVICurve(a=-1.0, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: invalid SVI parameters ...
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.parameters_are_valid():
            raise SVIError(
                f"invalid SVI parameters (a={self.a}, b={self.b}, "
                f"rho={self.rho}, m={self.m}, sigma={self.sigma}): "
                "need b >= 0, |rho| < 1, sigma > 0, all parameters finite, "
                "and a + b*sigma*sqrt(1-rho^2) >= 0"
            )

    def parameters_are_valid(self) -> bool:
        """Whether these parameters satisfy the arbitrage conditions.

        Always ``True`` for a constructed instance, since :meth:`__post_init__`
        rejects anything else. See :func:`check_parameters` to test raw values.

        >>> SVICurve(0.04, 0.10, -0.30, 0.0, 0.15).parameters_are_valid()
        True
        >>> SVICurve(0.04, -0.10, -0.30, 0.0, 0.15)
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: invalid SVI parameters ...
        """
        return check_parameters(self.a, self.b, self.rho, self.m, self.sigma)

    def minimum_total_variance(self) -> float:
        """Closed-form minimum of ``w(k)`` over all real ``k``.

        ``w`` attains its minimum where the derivative vanishes, at
        ``k = m - rho*sigma/sqrt(1 - rho^2)``, and the value there is exactly
        ``a + b*sigma*sqrt(1 - rho^2)``. Requiring that to be non-negative is
        what guarantees ``w(k) >= 0`` everywhere.

        >>> round(SVICurve(0.04, 0.10, 0.0, 0.0, 0.15).minimum_total_variance(), 10)
        0.055
        >>> round(SVICurve(0.04, 0.10, -0.5, 0.0, 0.15).minimum_total_variance(), 10)
        0.0529903811
        >>> round(SVICurve(0.04, 0.10, -0.3, 0.0, 0.15).minimum_total_variance(), 16)
        0.0543090880212542
        >>> SVICurve(0.04, 0.0, 0.0, 0.0, 0.15).minimum_total_variance()
        0.04
        """
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)

    def total_variance(self, k: float) -> float:
        """Total implied variance at log-moneyness ``k = ln(K/F)``.

        >>> curve = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> round(curve.total_variance(-0.20), 10)
        0.071
        >>> round(curve.total_variance(0.20), 10)
        0.059
        """
        shifted = k - self.m
        root = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.a + self.b * (self.rho * shifted + root)

    def total_variance_vector(self, k: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`total_variance`.

        >>> import numpy as np
        >>> curve = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> np.round(curve.total_variance_vector(np.array([-0.1, 0.0, 0.1])), 8)
        array([0.06102776, 0.055     , 0.05502776])
        """
        k_arr = np.asarray(k, dtype=float)
        shifted = k_arr - self.m
        root = np.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.a + self.b * (self.rho * shifted + root)

    def implied_volatility(self, k: float, maturity: float) -> float:
        """Implied volatility at log-moneyness ``k`` for the given maturity.

        Inverts ``total_variance = implied_vol^2 * maturity``.

        Raises:
            ValueError: If ``maturity`` is not positive.
            SVIError: If the total variance there is negative.

        >>> curve = SVICurve(0.04, 0.10, -0.30, 0.0, sigma=0.15)
        >>> round(curve.implied_volatility(0.0, 1.0), 10)
        0.234520788
        >>> round(curve.implied_volatility(0.0, 0.25), 10)
        0.469041576
        >>> curve.implied_volatility(0.0, 0.0)
        Traceback (most recent call last):
            ...
        ValueError: maturity must be positive, got 0.0
        """
        if maturity <= 0:
            raise ValueError(f"maturity must be positive, got {maturity}")
        variance = self.total_variance(k)
        if variance < 0:
            raise SVIError(f"negative total variance {variance} at k={k}")
        return math.sqrt(variance / maturity)

    def _d1(self, k: float) -> float:
        """First derivative of total variance with respect to ``k``."""
        shifted = k - self.m
        root = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.b * (self.rho + shifted / root)

    def _d1_vector(self, k: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`_d1`, used to locate the extrema of a difference."""
        k_arr = np.asarray(k, dtype=float)
        shifted = k_arr - self.m
        root = np.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.b * (self.rho + shifted / root)

    def _d2(self, k: float) -> float:
        """Second derivative of total variance with respect to ``k``."""
        shifted = k - self.m
        root = math.sqrt(shifted * shifted + self.sigma * self.sigma)
        return self.b * self.sigma * self.sigma / (root**3)

    def jump_wings_parameters(self, maturity: float) -> dict[str, float]:
        """SVI-Jump-Wings parameters, which practitioners can read directly.

        Raw SVI parameters have no trading interpretation. The SVI-JW
        reparameterisation introduced by Gatheral and Jacquier gives five
        numbers that mean something, which is why it is used in practice:

        ``atm_variance``
            implied variance at the money.
        ``atm_skew``
            d(sigma)/dk at k = 0, the quoted ATM volatility skew.
        ``put_wing_slope`` and ``call_wing_slope``
            slopes of the left and right wings.
        ``min_variance``
            the minimum implied variance over all strikes.

        These are the invariants across expiries when smiles scale with
        ``1/sqrt(w_t)``, so they are the natural way to check a term structure
        for consistency.

        Args:
            maturity: Time to expiry in years.

        Raises:
            ValueError: If ``maturity`` is not positive.

        >>> curve = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> jw = curve.jump_wings_parameters(1.0)
        >>> round(jw["atm_variance"], 10)
        0.055
        >>> round(jw["atm_skew"], 10)
        -0.0639602149
        >>> round(jw["min_variance"], 10)
        0.054309088
        >>> curve.jump_wings_parameters(0.0)
        Traceback (most recent call last):
            ...
        ValueError: maturity must be positive, got 0.0
        """
        if maturity <= 0:
            raise ValueError(f"maturity must be positive, got {maturity}")

        a, b, rho, m, sigma = self.a, self.b, self.rho, self.m, self.sigma
        root_m = math.sqrt(m * m + sigma * sigma)

        atm_variance = (a + b * (-rho * m + root_m)) / maturity
        sqrt_w = math.sqrt(atm_variance * maturity)

        return {
            "atm_variance": atm_variance,
            "atm_skew": (b / 2.0) * (-m / root_m + rho) / sqrt_w,
            "put_wing_slope": b * (1.0 - rho) / sqrt_w,
            "call_wing_slope": b * (1.0 + rho) / sqrt_w,
            "min_variance": (a + b * sigma * math.sqrt(1.0 - rho * rho)) / maturity,
        }

    def calendar_arbitrage_free(
        self, other: SVICurve, k_low: float = -4.0, k_high: float = 4.0, points: int = 2001
    ) -> bool:
        """Whether this slice and ``other`` never cross.

        Two SVI slices create calendar-spread arbitrage exactly when they
        intersect: at some log-moneyness the longer-dated total variance is
        below the shorter-dated one, so selling the long and buying the short
        is profitable. Gatheral and Jacquier reduce this to a quartic with no
        real root; this checks the same property directly by looking for a sign
        change in the difference, which is robust to the squaring steps their
        derivation needs and to the spurious roots those steps introduce.

        Equality of the two curves is permitted, so only a genuine sign change
        counts as an intersection. A tangential touch that does not change sign
        is therefore not reported.

        The search is numerical over ``[k_low, k_high]`` and is not a proof of
        anything outside that range. Inside it the result does not depend on
        ``points``: the grid is scanned for a sign change, and when it finds
        none, the stationary points of the difference are located from its
        analytic derivative and checked directly. That closes the gap a fixed
        grid leaves, where a narrow dip through zero or a crossing between two
        adjacent grid points would otherwise be missed. The extrema of a smooth
        function on a closed interval lie at its endpoints or at its stationary
        points, so checking those settles the question rather than sampling it.

        Args:
            other: The other slice, on the same underlying.
            k_low: Lower edge of the search range. A crossing outside it is not
                detected.
            k_high: Upper edge of the search range. A crossing outside it is not
                detected.
            points: Grid resolution for the initial scan.

        Raises:
            ValueError: If ``points`` is fewer than three, or if the search range
                is inverted.

        >>> short = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
        >>> long = SVICurve(0.09, 0.10, -0.30, 0.0, 0.15)
        >>> short.calendar_arbitrage_free(long)
        True

        A uniformly higher slice never crosses and is therefore fine: more
        total variance everywhere just means higher prices everywhere.

        Two slices cross when one is higher in the wings but lower near the
        money. Selling the richer and buying the cheaper is then profitable at
        some strike, which is calendar arbitrage:

        >>> skewed = SVICurve(0.05, 0.10, -0.70, 0.0, 0.15)
        >>> short.calendar_arbitrage_free(skewed)
        False
        >>> skewed.calendar_arbitrage_free(short)
        False

        A slice is never in calendar arbitrage with itself:

        >>> short.calendar_arbitrage_free(short)
        True

        The relation is symmetric, as it must be:

        >>> SVICurve(0.09, 0.10, -0.30, 0.0, 0.15).calendar_arbitrage_free(short)
        True

        A crossing just inside the boundary is still found, however coarse the
        initial grid:

        >>> edge = SVICurve(0.04, 0.10, -0.30, 3.98, 0.15)
        >>> short.calendar_arbitrage_free(edge, k_low=-4.0, k_high=4.0, points=3)
        False

        The same pair is clean when the crossing falls outside the searched
        range, which is the documented limit of a numerical check:

        >>> short.calendar_arbitrage_free(edge, k_low=-2.0, k_high=2.0, points=3)
        True
        """
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")
        if k_high <= k_low:
            raise ValueError(f"k_high ({k_high}) must be greater than k_low ({k_low})")

        k = np.linspace(k_low, k_high, points)
        difference = self.total_variance_vector(k) - other.total_variance_vector(k)

        # A sign change between adjacent grid points answers the question: the
        # curves cross somewhere in that cell, so they are not free.
        if np.any(difference[:-1] * difference[1:] < 0.0):
            return False

        # Otherwise look for extrema the grid may have stepped over. The
        # derivative of the difference is analytic, so each sign change in it
        # brackets exactly one stationary point, which bisection locates.
        #
        # This search runs on its own grid, independent of ``points``. Deriving
        # it from the caller's grid left a real gap: the derivative can be
        # positive only in a narrow band, so on a coarse grid both ends of a
        # cell sharing that band report the same sign, no sign change is seen,
        # and a genuine local extremum goes unexamined. Two slices whose
        # difference rose to +0.026 and fell to -0.55 inside a single cell were
        # reported free of calendar arbitrage. Missing an extremum here is a
        # false negative in an arbitrage check, which is the dangerous direction,
        # so the resolution of this scan is not left to the caller.
        search_points = max(points, _CALENDAR_SEARCH_MIN_POINTS)

        def slope(x: float) -> float:
            return float(self._d1_vector(np.array([x]))[0] - other._d1_vector(np.array([x]))[0])

        def value(x: float) -> float:
            return self.total_variance(x) - other.total_variance(x)

        search_k = np.linspace(k_low, k_high, search_points)
        d_slope = self._d1_vector(search_k) - other._d1_vector(search_k)
        candidates = [float(search_k[0]), float(search_k[-1])]

        for index in range(search_k.size - 1):
            left, right = float(d_slope[index]), float(d_slope[index + 1])
            if left == 0.0:
                candidates.append(float(search_k[index]))
            elif left * right < 0.0:
                low, high = float(search_k[index]), float(search_k[index + 1])
                for _ in range(60):
                    mid = 0.5 * (low + high)
                    if slope(low) * slope(mid) <= 0.0:
                        high = mid
                    else:
                        low = mid
                candidates.append(0.5 * (low + high))

        extrema = [value(x) for x in candidates]
        lowest, highest = min(extrema), max(extrema)

        # The curves are free unless the difference takes both signs. A tangent
        # that touches zero and turns back is permitted by the equality rule
        # above, and an exact touch is only ever zero up to rounding, so the
        # comparison carries a tolerance scaled to the size of the difference.
        scale = max(abs(lowest), abs(highest))
        tolerance = 1e-12 * scale if scale > 0.0 else 0.0
        return not (lowest < -tolerance and highest > tolerance)

    def butterfly_arbitrage_free(
        self, k_low: float = -2.0, k_high: float = 2.0, points: int = 401
    ) -> bool:
        """Check Gatheral's butterfly (density) condition across a strike range.

        Positive density requires, at every ``k``::

            (1 - k*w'(k)/(2*w(k)))^2 - w'(k)^2/4 * (1/w(k) + 1/4) + w''(k)/2 >= 0

        Where ``w(k)`` reaches zero or below, the condition fails by
        construction, since the first two terms are not defined there.

        This is a grid check, not a proof. It is the standard practical test, and
        a finer grid can only find more violations, never fewer.

        Raises:
            ValueError: If ``points`` is fewer than three.

        >>> SVICurve(0.04, 0.10, -0.30, 0.0, 0.15).butterfly_arbitrage_free()
        True
        >>> SVICurve(0.001, 2.0, 0.9, 0.0, 0.01).butterfly_arbitrage_free()
        False
        >>> SVICurve(0.04, 0.10, 0.0, 0.0, 0.15).butterfly_arbitrage_free(points=2)
        Traceback (most recent call last):
            ...
        ValueError: points must be at least 3, got 2
        """
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")

        k = np.linspace(k_low, k_high, points)
        w = self.total_variance_vector(k)

        if np.any(w <= 0):
            return False

        shifted = k - self.m
        root = np.sqrt(shifted * shifted + self.sigma * self.sigma)
        w_prime = self.b * (self.rho + shifted / root)
        w_double_prime = self.b * self.sigma**2 / root**3

        term_a = 1.0 - k * w_prime / (2.0 * w)
        term_b = 0.25 * w_prime**2 * (1.0 / w + 0.25)
        term_c = 0.5 * w_double_prime

        return bool(np.all(term_a**2 - term_b + term_c >= 0.0))


@dataclass(frozen=True, slots=True)
class SVIFit:
    """The result of fitting an SVI slice to quoted data."""

    curve: SVICurve
    rms_error: float
    strikes: tuple[float, ...]
    market_vols: tuple[float, ...]
    model_vols: tuple[float, ...]

    @property
    def max_error(self) -> float:
        """Largest absolute implied-volatility error across the fit.

        >>> import numpy as np
        >>> from black_scholes.surface import fit_svi
        >>> K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        >>> vols = np.array([0.28, 0.24, 0.22, 0.23, 0.25])
        >>> fit_svi(100.0, K, vols, 1.0).max_error < 0.01
        True
        """
        return max(abs(m - f) for m, f in zip(self.market_vols, self.model_vols, strict=True))


def _svi_residuals(
    params: np.ndarray, k: np.ndarray, w: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    a, b, rho, m, sigma = (float(x) for x in params)
    shifted = k - m
    root = np.sqrt(shifted * shifted + sigma * sigma)
    raw: np.ndarray = np.asarray(a + b * (rho * shifted + root) - w)
    residual: np.ndarray = scale * raw
    return residual


def fit_svi(
    forward: float,
    strikes: Sequence[float],
    vols: Sequence[float],
    maturity: float,
    *,
    check_butterfly: bool = True,
    max_vol_error: float | None = 0.10,
    weights: Literal["uniform", "vega"] = "vega",
) -> SVIFit:
    """Fit an SVI slice to implied volatilities.

    Fitting happens in total-variance space, where SVI is linear in ``a`` and
    far better conditioned than fitting volatilities directly. The problem is
    non-convex, so several starting points are tried and the cheapest result
    wins.

    Args:
        forward: Forward price ``S*e^((r-q)T)``.
        strikes: Strike prices, at least four.
        vols: Implied volatilities in decimal form matching ``strikes``.
        maturity: Time to expiry in years.
        check_butterfly: Reject fits failing the butterfly condition.
        max_vol_error: Largest tolerated absolute implied-volatility error at
            any quoted strike, in decimal form. SVI cannot represent every
            shape a market can quote; when it cannot, the optimiser returns a
            curve that passes the arbitrage checks but misses the data badly.
            This bound is what stops such a result being reported as a
            successful fit. A real single-expiry smile fits to well under one
            volatility point, so the default of 0.10 is already loose.
        weights: How strikes are weighted in the least-squares objective.
            ``"vega"`` (the default) weights each strike by its Black-Scholes
            vega, so ATM and near-ATM quotes — where the price estimate is
            most sensitive — dominate, and deep out-of-the-money crus with
            wide, unstable implied-volatility errors do not skew the fit.
            ``"uniform"`` weights every strike equally, which was the only
            behaviour before this option existed. The reported
            ``SVIFit.rms_error`` and ``SVIFit.max_error`` are always measured
            in raw implied-volatility terms, regardless of this choice.

    Returns:
        An :class:`SVIFit` holding the curve and its errors.

    Raises:
        SVIError: If inputs are malformed, no arbitrage-free fit exists, or the
            best fit still misses by more than ``max_vol_error``.

    A flat slice is recovered exactly:

    >>> import numpy as np
    >>> K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    >>> fit = fit_svi(100.0, K, np.full(5, 0.20), 1.0)
    >>> round(fit.rms_error, 12)
    0.0
    >>> all(abs(v - 0.20) < 1e-6 for v in fit.model_vols)
    True

    An arbitrage-free smile generated from SVI is recovered to machine
    precision, parameters included:

    >>> truth = SVICurve(0.04, 0.10, -0.30, 0.0, 0.15)
    >>> vols = [truth.implied_volatility(math.log(k / 100.0), 1.0) for k in K]
    >>> fit = fit_svi(100.0, K, vols, 1.0)
    >>> round(fit.rms_error, 12)
    0.0
    >>> round(fit.curve.a, 10), round(fit.curve.b, 10), round(fit.curve.rho, 10)
    (0.04, 0.1, -0.3)

    A smile steep enough that no SVI curve can match it without a negative
    density is rejected rather than fitted:

    >>> steep = np.array([0.40, 0.30, 0.23, 0.19, 0.23, 0.28, 0.36])
    >>> wide = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0])
    >>> fit_svi(100.0, wide, steep, 1.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: SVI fit did not converge to an arbitrage-free parameter set: fitted curve violates the butterfly condition
    >>> fit_svi(100.0, wide, steep, 1.0, check_butterfly=False).curve.b > 0
    True

    SVI cannot fit every shape a market can quote. A smile with a spike in the
    middle is representable by no SVI curve, so the optimiser returns a curve
    that passes the arbitrage tests yet misses the data badly. The error
    tolerance is the backstop for exactly that case:

    >>> spike = np.array([0.20, 0.22, 0.40, 0.60, 0.40, 0.22, 0.20])
    >>> strikes7 = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0])
    >>> fit_svi(100.0, strikes7, spike, 1.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: SVI cannot fit this smile: worst error ...

    Lifting the tolerance reveals the fit the guard was hiding, and it is
    arbitrage-free, which is why an error bound is needed alongside the
    butterfly check rather than instead of it:

    >>> loose_fit = fit_svi(100.0, strikes7, spike, 1.0, max_vol_error=None)
    >>> loose_fit.max_error > 0.2
    True
    >>> loose_fit.curve.parameters_are_valid()
    True

    The tolerance applies to ordinary mismatches too:

    >>> loose = np.array([70.0, 90.0, 100.0, 110.0, 150.0])
    >>> smile = np.array([0.30, 0.25, 0.20, 0.21, 0.28])
    >>> fit_svi(100.0, loose, smile, 1.0, max_vol_error=0.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: SVI cannot fit this smile: worst error ...

    Too few points is an error, not a meaningless fit:

    >>> fit_svi(100.0, [90.0, 100.0, 110.0], [0.2, 0.2, 0.2], 1.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: need at least 4 strikes to fit an SVI slice, got 3
    """
    k_arr = np.asarray(strikes, dtype=float)
    v_arr = np.asarray(vols, dtype=float)

    if k_arr.ndim != 1 or v_arr.ndim != 1:
        raise SVIError("strikes and vols must be one-dimensional")
    if k_arr.size != v_arr.size:
        raise SVIError(f"strikes and vols differ in length: {k_arr.size} vs {v_arr.size}")
    if k_arr.size < 4:
        raise SVIError(f"need at least 4 strikes to fit an SVI slice, got {k_arr.size}")
    if not math.isfinite(forward):
        raise SVIError(f"forward must be a finite number, got {forward}")
    if forward <= 0:
        raise SVIError(f"forward must be positive, got {forward}")
    if not math.isfinite(maturity):
        raise SVIError(f"maturity must be a finite number, got {maturity}")
    if maturity <= 0:
        raise SVIError(f"maturity must be positive, got {maturity}")
    if not np.all(np.isfinite(k_arr)):
        raise SVIError("strikes must all be finite numbers")
    if not np.all(np.isfinite(v_arr)):
        raise SVIError("implied volatilities must all be finite numbers")
    if np.any(k_arr <= 0):
        raise SVIError("strikes must be positive")
    if np.any(v_arr <= 0):
        raise SVIError("implied volatilities must be positive")

    k = np.log(k_arr / forward)
    w = v_arr**2 * maturity

    # A flat slice is the degenerate case the parameter bounds cannot express,
    # so handle it directly instead of letting the optimiser fail.
    if float(np.ptp(w)) <= 1e-12:
        level = float(np.mean(w))
        a = level if level > 0 else 1e-8
        return SVIFit(
            curve=SVICurve(a=a, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rms_error=0.0,
            strikes=tuple(float(x) for x in k_arr),
            market_vols=tuple(float(x) for x in v_arr),
            model_vols=tuple(float(x) for x in v_arr),
        )

    bounds = (
        np.array([-10.0, 0.0, -0.999, -2.0, 1e-4]),
        np.array([10.0, 5.0, 0.999, 2.0, 3.0]),
    )

    # Compute weights for the least-squares objective.
    # Vega weights are based on Black-Scholes vega at the initial mean volatility.
    # This gives more weight to ATM/near-ATM strikes where implied vol is
    # more precisely measured, and less weight to deep OTM/ITM strikes.
    if weights == "vega":
        # Use initial mean volatility as reference for vega weights
        v_init = float(np.mean(v_arr))
        # Black-Scholes vega for each strike: proportional to forward * sqrt(T) * phi(d1)
        # d1 = (log(F/K) + 0.5 * v^2 * T) / (v * sqrt(T))
        # vega is proportional to forward * sqrt(maturity) * phi(d1)
        sqrt_T = math.sqrt(maturity)
        log_moneyness = np.log(forward / k_arr)
        d1 = (log_moneyness + 0.5 * v_init * v_init * maturity) / (v_init * sqrt_T)
        # Standard normal PDF: phi(d1) = exp(-d1^2/2) / sqrt(2*pi)
        phi_d1 = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        # Vega weight proportional to forward * sqrt(T) * phi(d1)
        # Normalize so mean weight is 1 to keep cost function scale similar
        scale = forward * sqrt_T * phi_d1
        scale = scale / float(np.mean(scale))
    else:
        scale = np.ones_like(k_arr)

    best_cost = math.inf
    best_params: np.ndarray | None = None

    for b0 in (0.0, 0.05, 0.15, 0.4):
        for m0 in (-0.3, 0.0, 0.3):
            for rho0 in (-0.5, 0.0, 0.5):
                guess = np.clip(
                    np.array([float(np.mean(w)), b0, rho0, m0, 0.2]),
                    bounds[0] + 1e-9,
                    bounds[1] - 1e-9,
                )
                try:
                    result = least_squares(
                        _svi_residuals,
                        guess,
                        bounds=bounds,
                        args=(k, w, scale),
                        method="trf",
                        max_nfev=20000,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not result.success:
                    continue
                cost = float(np.sum(result.fun**2))
                if cost < best_cost:
                    best_cost = cost
                    best_params = result.x

    if best_params is None:
        raise SVIError(SVI_FIT_FAILURE)

    a, b, rho, m, sigma = (float(x) for x in best_params)

    # The optimiser is not constrained to the arbitrage region, so repair the
    # minimum-variance condition if the fit landed just outside it.
    root = math.sqrt(1.0 - rho * rho)
    if a + b * sigma * root < 0:
        a = max(0.0, -b * sigma * root + 1e-12)

    try:
        curve = SVICurve(a=a, b=max(b, 0.0), rho=rho, m=m, sigma=max(sigma, 1e-4))
    except SVIError as exc:
        raise SVIError(f"{SVI_FIT_FAILURE}: {exc}") from exc

    if check_butterfly and not curve.butterfly_arbitrage_free():
        raise SVIError(f"{SVI_FIT_FAILURE}: fitted curve violates the butterfly condition")

    model_vols = tuple(curve.implied_volatility(float(ki), maturity) for ki in k)
    errors = [m - f for m, f in zip(v_arr, model_vols, strict=True)]
    rms = float(math.sqrt(sum(e * e for e in errors) / len(errors)))
    worst = float(max(abs(e) for e in errors))

    if max_vol_error is not None and worst > max_vol_error:
        worst_strike = int(np.argmax(np.abs(errors)))
        raise SVIError(
            f"SVI cannot fit this smile: worst error {worst:.2%} at strike "
            f"{float(k_arr[worst_strike]):g}, tolerance {max_vol_error:.2%}. "
            "SVI cannot represent every quoted shape; widen --relaxed to inspect."
        )

    return SVIFit(
        curve=curve,
        rms_error=rms,
        strikes=tuple(float(x) for x in k_arr),
        market_vols=tuple(float(x) for x in v_arr),
        model_vols=model_vols,
    )


def _validate_surface_quotes(strikes: Sequence[float], implied_vols: Sequence[float]) -> None:
    """Validate the quoted strikes and volatilities of a smile.

    Shared by :meth:`VolSurface.__post_init__`, which every construction path
    goes through, and by :meth:`VolSurface.from_market_prices`, which calls it
    up front so a malformed quote is rejected before any price inversion runs.

    The four-strike minimum that SVI fitting needs is deliberately *not* checked
    here. That is a property of the fitter, not of a smile: a single quote is a
    perfectly valid surface to interpolate.

    Raises:
        SVIError: If the collections are empty, differ in length, contain
            duplicate or non-positive strikes, or contain non-finite or
            non-positive volatilities.
    """
    strike_list = [float(k) for k in strikes]
    vol_list = [float(v) for v in implied_vols]

    if not strike_list:
        raise SVIError(f"VolSurface needs at least one quoted strike, got {len(strike_list)}")
    if len(strike_list) != len(vol_list):
        raise SVIError(
            f"strikes and implied_vols differ in length: {len(strike_list)} vs {len(vol_list)}"
        )

    _validate_strike_list(strike_list)
    _validate_vol_list(vol_list)


def _validate_strike_list(strikes: Sequence[float]) -> None:
    """Validate quoted strikes: at least one, positive, finite, and distinct.

    Raises:
        SVIError: If any of those conditions is violated.
    """
    strike_list = [float(k) for k in strikes]
    if not strike_list:
        raise SVIError("VolSurface needs at least one quoted strike, got 0")

    seen: set[float] = set()
    for strike in strike_list:
        if not math.isfinite(strike):
            raise SVIError(f"strikes must be finite numbers, got {strike}")
        if strike <= 0:
            raise SVIError(f"strikes must be positive, got {strike}")
        if strike in seen:
            raise SVIError(f"duplicate strikes make interpolation ambiguous: {strike}")
        seen.add(strike)


def _validate_vol_list(vols: Sequence[float]) -> None:
    """Validate quoted implied volatilities: positive and finite.

    Raises:
        SVIError: If any of those conditions is violated.
    """
    for vol in (float(v) for v in vols):
        if not math.isfinite(vol):
            raise SVIError(f"implied volatilities must be finite numbers, got {vol}")
        if vol <= 0:
            raise SVIError(f"implied volatilities must be positive, got {vol}")


@dataclass(frozen=True, slots=True)
class VolSurface:
    """An implied-volatility smile built from market option prices.

    Prices are inverted with the closed-form solver, then a single expiry is
    fitted with SVI. Multi-expiry surfaces are built by calling
    :meth:`from_market_prices` per expiry; calendar arbitrage across expiries is
    the caller's check, not this class's.

    >>> surface = VolSurface.from_market_prices(
    ...     spot=100.0,
    ...     strikes=[70.0, 90.0, 100.0, 110.0, 150.0],
    ...     prices=[34.3824, 18.0797, 11.7507, 7.4028, 1.3415],
    ...     time_to_maturity=1.0,
    ...     risk_free_rate=0.05,
    ... )
    >>> round(surface.implied_volatility(100.0), 8)
    0.23451999
    >>> surface.fit().max_error < 1e-8
    True

    Direct construction is validated too, so an empty or malformed surface fails
    at construction instead of raising ``IndexError`` later during
    interpolation. A single quote is a valid surface: interpolation needs only
    one point, and the four-strike minimum belongs to SVI fitting, not here.

    >>> VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, 0.0).implied_volatility(100.0)
    0.2
    >>> VolSurface(100.0, (), (), 1.0, 0.05, 0.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: VolSurface needs at least one quoted strike, got 0
    >>> VolSurface(100.0, (90.0, 100.0), (0.2,), 1.0, 0.05, 0.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: strikes and implied_vols differ in length: 2 vs 1
    >>> VolSurface(100.0, (100.0, 100.0), (0.2, 0.3), 1.0, 0.05, 0.0)
    Traceback (most recent call last):
        ...
    black_scholes.surface.SVIError: duplicate strikes make interpolation ambiguous: 100.0
    """

    spot: float
    strikes: tuple[float, ...]
    implied_vols: tuple[float, ...]
    time_to_maturity: float
    risk_free_rate: float
    dividend_yield: float
    option_type: OptionType = OptionType.CALL

    def __post_init__(self) -> None:
        _validate_surface_quotes(self.strikes, self.implied_vols)
        checks = (
            ("spot", self.spot, True),
            ("time_to_maturity", self.time_to_maturity, True),
            ("risk_free_rate", self.risk_free_rate, False),
            ("dividend_yield", self.dividend_yield, False),
        )
        for name, value, strictly_positive in checks:
            if not math.isfinite(value):
                raise SVIError(f"{name} must be a finite number, got {value}")
            if strictly_positive and value <= 0:
                raise SVIError(f"{name} must be positive, got {value}")
            if not strictly_positive and value < 0:
                raise SVIError(f"{name} must be non-negative, got {value}")
        if not isinstance(self.option_type, OptionType):
            raise SVIError(f"option_type must be OptionType, got {type(self.option_type).__name__}")

    @property
    def forward(self) -> float:
        """Forward price of the underlying, ``S*e^((r-q)T)``.

        >>> round(VolSurface(100.0, (100.0,), (0.2,), 1.0, 0.05, 0.0).forward, 6)
        105.12711
        """
        return self.spot * math.exp(
            (self.risk_free_rate - self.dividend_yield) * self.time_to_maturity
        )

    @classmethod
    def from_market_prices(
        cls,
        spot: float,
        strikes: list[float] | tuple[float, ...],
        prices: list[float] | tuple[float, ...],
        time_to_maturity: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL,
    ) -> VolSurface:
        """Build a smile by inverting market prices to implied volatilities.

        Raises:
            SVIError: If lengths differ or any price breaks no-arbitrage bounds.

        >>> VolSurface.from_market_prices(
        ...     100.0, [90.0, 100.0], [18.08, 11.75], 1.0, 0.05
        ... ).strikes
        (90.0, 100.0)

        A price outside the no-arbitrage bounds is reported with its strike,
        so a bad quote is identifiable rather than just rejected:

        >>> try:
        ...     VolSurface.from_market_prices(100.0, [90.0], [0.5], 1.0, 0.05)
        ... except SVIError as exc:
        ...     print(exc)
        strike 90.0: cannot invert price 0.5: market price 0.500000 outside no-arbitrage bounds [14.389352, 100.000000]

        Mismatched lengths are caught before any inversion runs:

        >>> VolSurface.from_market_prices(100.0, [90.0, 100.0], [18.08], 1.0, 0.05)
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: strikes and prices differ in length: 2 vs 1

        Strike and price validity is checked up front too, so a bad quote is
        named rather than surfacing later from inside the inversion:

        >>> VolSurface.from_market_prices(100.0, [90.0, 90.0], [18.08, 19.0], 1.0, 0.05)
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: duplicate strikes make interpolation ambiguous: 90.0
        >>> VolSurface.from_market_prices(100.0, [90.0], [float("nan")], 1.0, 0.05)
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: market prices must be finite numbers, got nan
        >>> VolSurface.from_market_prices(100.0, [float("inf")], [18.08], 1.0, 0.05)
        Traceback (most recent call last):
            ...
        black_scholes.surface.SVIError: strikes must be finite numbers, got inf
        """
        strike_list = [float(k) for k in strikes]
        price_list = [float(p) for p in prices]
        if len(strike_list) != len(price_list):
            raise SVIError(
                f"strikes and prices differ in length: {len(strike_list)} vs {len(price_list)}"
            )

        # Validate the quotes before inverting anything, so a malformed strike
        # is reported directly instead of as a confusing inversion failure.
        _validate_strike_list(strike_list)
        for price in price_list:
            if not math.isfinite(price):
                raise SVIError(f"market prices must be finite numbers, got {price}")

        vols: list[float] = []
        for strike, price in zip(strike_list, price_list, strict=True):
            params = OptionParams(
                spot=spot,
                strike=strike,
                time_to_maturity=time_to_maturity,
                risk_free_rate=risk_free_rate,
                volatility=0.2,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )
            try:
                vols.append(implied_volatility(price, params))
            except ValueError as exc:
                raise SVIError(f"strike {strike}: cannot invert price {price}: {exc}") from exc

        return cls(
            spot=spot,
            strikes=tuple(strike_list),
            implied_vols=tuple(vols),
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )

    def implied_volatility(self, strike: float) -> float:
        """Implied volatility at ``strike``, linearly interpolated in log-moneyness.

        This interpolates the *quoted* points. Use :meth:`fit` then
        :meth:`SVICurve.implied_volatility` for a smooth arbitrage-aware
        extrapolation off the quoted strikes.

        >>> surface = VolSurface.from_market_prices(
        ...     100.0, [70.0, 90.0, 100.0, 110.0, 150.0],
        ...     [34.3824, 18.0797, 11.7507, 7.4028, 1.3415], 1.0, 0.05,
        ... )
        >>> round(surface.implied_volatility(90.0), 8)
        0.24797458
        >>> round(surface.implied_volatility(100.0), 8)
        0.23451999
        """
        forward = self.forward
        target = math.log(float(strike) / forward)

        points = sorted(
            zip(
                (math.log(k / forward) for k in self.strikes),
                self.implied_vols,
                strict=True,
            )
        )

        if target <= points[0][0]:
            return points[0][1]
        if target >= points[-1][0]:
            return points[-1][1]

        for (k_lo, v_lo), (k_hi, v_hi) in itertools.pairwise(points):
            if k_lo <= target <= k_hi:
                if k_hi == k_lo:
                    return v_lo
                weight = (target - k_lo) / (k_hi - k_lo)
                return v_lo + weight * (v_hi - v_lo)

        return points[-1][1]

    def fit(
        self,
        *,
        check_butterfly: bool = True,
        max_vol_error: float | None = 0.10,
        weights: Literal["uniform", "vega"] = "vega",
    ) -> SVIFit:
        """Fit an SVI slice to this smile's implied volatilities.

        Args:
            check_butterfly: Reject a fit that violates the density condition.
            max_vol_error: Largest tolerated error at any quoted strike.
                ``None`` disables the bound.
            weights: How strikes are weighted in the least-squares objective.
                ``"vega"`` (default) weights by Black-Scholes vega; ``"uniform"``
                weights all strikes equally.

        Raises:
            SVIError: If fewer than four strikes are quoted, or no acceptable
                fit exists.

        >>> surface = VolSurface.from_market_prices(
        ...     100.0, [70.0, 90.0, 100.0, 110.0, 150.0],
        ...     [34.3824, 18.0797, 11.7507, 7.4028, 1.3415], 1.0, 0.05,
        ... )
        >>> round(surface.fit().rms_error, 12)
        0.0
        """
        return fit_svi(
            self.forward,
            self.strikes,
            self.implied_vols,
            self.time_to_maturity,
            check_butterfly=check_butterfly,
            max_vol_error=max_vol_error,
            weights=weights,
        )

    def fit_relaxed(self, *, weights: Literal["uniform", "vega"] = "vega") -> SVIFit:
        """Fit without the butterfly or error-tolerance guards.

        Lets a rejected fit be inspected rather than silently dropped, so the
        rejection is visible.

        Args:
            weights: How strikes are weighted in the least-squares objective.
                ``"vega"`` (default) weights by Black-Scholes vega; ``"uniform"``
                weights all strikes equally.

        >>> surface = VolSurface.from_market_prices(
        ...     100.0, [70.0, 90.0, 100.0, 110.0, 150.0],
        ...     [34.3824, 18.0797, 11.7507, 7.4028, 1.3415], 1.0, 0.05,
        ... )
        >>> surface.fit_relaxed().curve.parameters_are_valid()
        True
        """
        return self.fit(check_butterfly=False, max_vol_error=None, weights=weights)
