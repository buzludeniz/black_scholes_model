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


def check_parameters(a: float, b: float, rho: float, m: float, sigma: float) -> bool:
    """Whether raw SVI parameters satisfy the closed-form arbitrage conditions.

    Exposed separately from :meth:`SVICurve.parameters_are_valid` because the
    constructor rejects invalid parameters, so an instance can never report
    ``False``. The optimiser does produce invalid parameters, and this is what
    tests them.

    The conditions are ``b >= 0``, ``|rho| < 1``, ``sigma > 0``, and a
    non-negative minimum total variance.

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
    """
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
                "need b >= 0, |rho| < 1, sigma > 0, and a + b*sigma*sqrt(1-rho^2) >= 0"
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
        if self.b < 0:
            return False
        if not -1.0 < self.rho < 1.0:
            return False
        if self.sigma <= 0:
            return False
        return self.minimum_total_variance() >= 0.0

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
        counts as an intersection.

        Args:
            other: The other slice, on the same underlying.
            k_low: Lower edge of the search range.
            k_high: Upper edge of the search range.
            points: Grid resolution.

        Raises:
            ValueError: If ``points`` is fewer than three.

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
        """
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")

        k = np.linspace(k_low, k_high, points)
        difference = self.total_variance_vector(k) - other.total_variance_vector(k)

        return bool(np.all(difference >= 0.0) or np.all(difference <= 0.0))

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


def _svi_residuals(params: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = (float(x) for x in params)
    shifted = k - m
    root = np.sqrt(shifted * shifted + sigma * sigma)
    residual: np.ndarray = np.asarray(a + b * (rho * shifted + root) - w)
    return residual


def fit_svi(
    forward: float,
    strikes: Sequence[float],
    vols: Sequence[float],
    maturity: float,
    *,
    check_butterfly: bool = True,
    max_vol_error: float | None = 0.10,
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
    if forward <= 0:
        raise SVIError(f"forward must be positive, got {forward}")
    if maturity <= 0:
        raise SVIError(f"maturity must be positive, got {maturity}")
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
                        args=(k, w),
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
    """

    spot: float
    strikes: tuple[float, ...]
    implied_vols: tuple[float, ...]
    time_to_maturity: float
    risk_free_rate: float
    dividend_yield: float
    option_type: OptionType = OptionType.CALL

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
        """
        strike_list = [float(k) for k in strikes]
        price_list = [float(p) for p in prices]
        if len(strike_list) != len(price_list):
            raise SVIError(
                f"strikes and prices differ in length: {len(strike_list)} vs {len(price_list)}"
            )

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

    def fit(self, *, check_butterfly: bool = True, max_vol_error: float | None = 0.10) -> SVIFit:
        """Fit an SVI slice to this smile's implied volatilities.

        Args:
            check_butterfly: Reject a fit that violates the density condition.
            max_vol_error: Largest tolerated error at any quoted strike.
                ``None`` disables the bound.

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
        )

    def fit_relaxed(self) -> SVIFit:
        """Fit without the butterfly or error-tolerance guards.

        Lets a rejected fit be inspected rather than silently dropped, so the
        rejection is visible.

        >>> surface = VolSurface.from_market_prices(
        ...     100.0, [70.0, 90.0, 100.0, 110.0, 150.0],
        ...     [34.3824, 18.0797, 11.7507, 7.4028, 1.3415], 1.0, 0.05,
        ... )
        >>> surface.fit_relaxed().curve.parameters_are_valid()
        True
        """
        return self.fit(check_butterfly=False, max_vol_error=None)
