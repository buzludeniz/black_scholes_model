"""
Black-Scholes Option Pricing Model.

Implements the Black-Scholes-Merton formula for European option pricing
along with all major Greeks, implied volatility inversion, and a
Monte Carlo reference implementation for validation.

Every public function carries doctests, so the examples in the docstrings
are executable and verified by the test suite:

    python -m pytest --doctest-modules src/black_scholes
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm

__version__ = "0.2.0"

# Trading days per year, used to express Theta per calendar day.
DAYS_PER_YEAR = 365.0

# Percent scale used for reporting Vega and Rho (per 1% move).
PERCENT = 0.01


class OptionType(Enum):
    """European option type.

    >>> OptionType("call") is OptionType.CALL
    True
    >>> OptionType.CALL.value
    'call'
    """

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class OptionParams:
    """Parameters for the Black-Scholes model.

    Args:
        spot: Current underlying price (S).
        strike: Strike price (K).
        time_to_maturity: Time to expiration in years (T).
        risk_free_rate: Continuously compounded risk-free rate (r).
        volatility: Annualized volatility (sigma).
        option_type: Call or put.
        dividend_yield: Continuous dividend yield (q).

    Raises:
        ValueError: If any input is not finite, or violates a positivity
            constraint. ``NaN`` and ``+/-inf`` are rejected for every numeric
            field, because comparisons such as ``value <= 0`` are false for
            ``NaN`` and would otherwise let it reach the pricing formulas.

    >>> p = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20)
    >>> p.spot, p.strike, p.volatility
    (100.0, 100.0, 0.2)
    >>> p.option_type
    <OptionType.CALL: 'call'>
    >>> OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT).option_type
    <OptionType.PUT: 'put'>
    >>> OptionParams(-1.0, 100.0, 1.0, 0.05, 0.20)
    Traceback (most recent call last):
        ...
    ValueError: spot must be positive, got -1.0
    >>> OptionParams(100.0, 100.0, 1.0, 0.05, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: volatility must be positive, got 0.0

    ``NaN`` and infinities are rejected for every numeric field:

    >>> OptionParams(float("nan"), 100.0, 1.0, 0.05, 0.20)
    Traceback (most recent call last):
        ...
    ValueError: spot must be a finite number, got nan
    >>> OptionParams(100.0, float("inf"), 1.0, 0.05, 0.20)
    Traceback (most recent call last):
        ...
    ValueError: strike must be a finite number, got inf
    >>> OptionParams(100.0, 100.0, float("-inf"), 0.05, 0.20)
    Traceback (most recent call last):
        ...
    ValueError: time_to_maturity must be a finite number, got -inf
    >>> OptionParams(100.0, 100.0, 1.0, 0.05, float("nan"))
    Traceback (most recent call last):
        ...
    ValueError: volatility must be a finite number, got nan
    >>> OptionParams(100.0, 100.0, 1.0, float("nan"), 0.20)
    Traceback (most recent call last):
        ...
    ValueError: risk_free_rate must be a finite number, got nan
    >>> OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL, float("nan"))
    Traceback (most recent call last):
        ...
    ValueError: dividend_yield must be a finite number, got nan

    Negative rates and dividend yields stay rejected, as before:

    >>> OptionParams(100.0, 100.0, 1.0, -0.01, 0.20)
    Traceback (most recent call last):
        ...
    ValueError: risk_free_rate must be non-negative, got -0.01
    """

    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    option_type: OptionType = OptionType.CALL
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        checks = (
            ("spot", self.spot, True),
            ("strike", self.strike, True),
            ("time_to_maturity", self.time_to_maturity, True),
            ("volatility", self.volatility, True),
            ("risk_free_rate", self.risk_free_rate, False),
            ("dividend_yield", self.dividend_yield, False),
        )
        for name, value, strictly_positive in checks:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
            if strictly_positive and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if not strictly_positive and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not isinstance(self.option_type, OptionType):
            raise TypeError(
                f"option_type must be OptionType, got {type(self.option_type).__name__}"
            )

    @property
    def forward_price(self) -> float:
        """Forward price of the underlying (S·e^(-qT)/e^(-rT)).

        >>> round(OptionParams(100.0, 100.0, 1.0, 0.05, 0.20).forward_price, 5)
        105.12711
        """
        return self.spot * math.exp(
            (self.risk_free_rate - self.dividend_yield) * self.time_to_maturity
        )

    def replace_volatility(self, volatility: float) -> OptionParams:
        """Return a copy with a different volatility.

        >>> p = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20)
        >>> p.replace_volatility(0.35).volatility
        0.35
        >>> p.volatility
        0.2
        """
        return OptionParams(
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            risk_free_rate=self.risk_free_rate,
            volatility=volatility,
            option_type=self.option_type,
            dividend_yield=self.dividend_yield,
        )


@dataclass(frozen=True, slots=True)
class Greeks:
    """Option Greeks, quoted in trader-friendly units.

    Vega is per 1% volatility move, Theta is per calendar day, and Rho is
    per 1% rate move.

    >>> g = Greeks(0.5, 0.02, 0.30, -0.05, 0.40)
    >>> g.as_dict()["delta"]
    0.5
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        """Return the Greeks as a plain dict."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Complete pricing result with Greeks.

    >>> r = price_option(OptionParams(100.0, 100.0, 1.0, 0.05, 0.20))
    >>> round(r.price, 6)
    10.450584
    >>> "Delta" in str(r)
    True
    """

    price: float
    greeks: Greeks
    params: OptionParams

    def __str__(self) -> str:
        g = self.greeks
        return (
            f"Price: {self.price:.6f}\n"
            f"Delta: {g.delta:.6f}\n"
            f"Gamma: {g.gamma:.6f}\n"
            f"Vega:  {g.vega:.6f}\n"
            f"Theta: {g.theta:.6f}\n"
            f"Rho:   {g.rho:.6f}"
        )


def _d1_d2(params: OptionParams) -> tuple[float, float]:
    """Compute the d1 and d2 terms of the Black-Scholes formula.

    >>> tuple(round(v, 6) for v in _d1_d2(OptionParams(100.0, 100.0, 1.0, 0.05, 0.20)))
    (0.35, 0.15)
    """
    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    vol, q = params.volatility, params.dividend_yield
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    return d1, d2


def black_scholes_price(params: OptionParams) -> float:
    """Price a European option with the Black-Scholes-Merton formula.

    Call: ``S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)``
    Put:  ``K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)``

    Returns a native Python ``float``.

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> round(black_scholes_price(call), 6)
    10.450584
    >>> round(black_scholes_price(put), 6)
    5.573526
    >>> type(black_scholes_price(call)) is float
    True

    Put-call parity holds: ``C - P == S·e^(-qT) - K·e^(-rT)``. A continuous
    dividend yield of 3% pulls that spread down to 1.921611:

    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> c3 = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL, 0.03)
    >>> p3 = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT, 0.03)
    >>> round(black_scholes_price(call) - black_scholes_price(put), 6)
    4.877058
    >>> round(black_scholes_price(c3) - black_scholes_price(p3), 6)
    1.921611

    A deep in-the-money call converges to its discounted intrinsic value:

    >>> deep = OptionParams(200.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> round(black_scholes_price(deep), 4)
    104.8777
    """
    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    q = params.dividend_yield
    d1, d2 = _d1_d2(params)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if params.option_type is OptionType.CALL:
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    # Guard against tiny negative values from floating point cancellation.
    return float(max(price, 0.0))


def black_scholes_greeks(params: OptionParams) -> Greeks:
    """Compute the option Greeks analytically.

    Units: Delta and Gamma are absolute; Vega is per 1% volatility move;
    Theta is per calendar day; Rho is per 1% rate move.

    >>> p = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> g = black_scholes_greeks(p)
    >>> round(g.delta, 6), round(g.gamma, 6)
    (0.636831, 0.018762)
    >>> round(g.vega, 6), round(g.theta, 6), round(g.rho, 6)
    (0.37524, -0.017573, 0.532325)

    Gamma and Vega are identical for calls and puts; Delta, Theta and Rho
    differ:

    >>> pc = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> pp = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> gc, gp = black_scholes_greeks(pc), black_scholes_greeks(pp)
    >>> round(gc.gamma - gp.gamma, 12), round(gc.vega - gp.vega, 12)
    (0.0, 0.0)
    >>> round(gc.delta - gp.delta, 6)
    1.0
    >>> gc.rho > 0 > gp.rho
    True
    """
    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    vol, q = params.volatility, params.dividend_yield
    d1, d2 = _d1_d2(params)

    sqrt_T = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = float(norm.pdf(d1))

    gamma = disc_q * pdf_d1 / (S * vol * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T * PERCENT

    if params.option_type is OptionType.CALL:
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * vol / (2 * sqrt_T)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        ) / DAYS_PER_YEAR
        rho = K * T * disc_r * norm.cdf(d2) * PERCENT
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -S * disc_q * pdf_d1 * vol / (2 * sqrt_T)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        ) / DAYS_PER_YEAR
        rho = -K * T * disc_r * norm.cdf(-d2) * PERCENT

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def price_option(params: OptionParams) -> PricingResult:
    """Price an option and return the price together with its Greeks.

    >>> r = price_option(OptionParams(100.0, 100.0, 1.0, 0.05, 0.20))
    >>> round(r.price, 4)
    10.4506
    >>> r.params is r.params
    True
    >>> round(r.greeks.delta, 4)
    0.6368
    """
    return PricingResult(
        price=black_scholes_price(params),
        greeks=black_scholes_greeks(params),
        params=params,
    )


def arbitrage_bounds(params: OptionParams) -> tuple[float, float]:
    """Return the no-arbitrage price bounds ``(lower, upper)``.

    Lower bound is discounted intrinsic; upper bound is the discounted
    forward for calls, and the discounted strike for puts.

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> lo, hi = arbitrage_bounds(call)
    >>> round(lo, 6), round(hi, 6)
    (4.877058, 100.0)
    >>> lo, hi = arbitrage_bounds(put)
    >>> round(lo, 6), round(hi, 6)
    (0.0, 95.122942)

    Any model price must land inside these bounds:

    >>> call = OptionParams(150.0, 100.0, 0.5, 0.03, 0.40, OptionType.CALL)
    >>> lo, hi = arbitrage_bounds(call)
    >>> lo <= black_scholes_price(call) <= hi
    True
    """
    T, r, q = params.time_to_maturity, params.risk_free_rate, params.dividend_yield
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if params.option_type is OptionType.CALL:
        lower = max(params.spot * disc_q - params.strike * disc_r, 0.0)
        upper = params.spot * disc_q
    else:
        lower = max(params.strike * disc_r - params.spot * disc_q, 0.0)
        upper = params.strike * disc_r
    return float(lower), float(upper)


def implied_volatility(
    market_price: float,
    params: OptionParams,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
) -> float:
    """Recover the volatility implied by an observed option price.

    Uses bisection, which is unconditionally stable because the Black-Scholes
    price is monotonic in volatility. A Newton-Raphson refinement runs first
    to get close, then the bisection bracket guarantees the final answer.

    Args:
        market_price: Observed option price.
        params: Option parameters. ``volatility`` is ignored.
        tol: Absolute tolerance on the implied volatility.
        max_iter: Maximum bisection iterations.
        vol_lower: Lower edge of the volatility search bracket.
        vol_upper: Upper edge of the volatility search bracket.

    Returns:
        Annualized implied volatility as a native ``float``.

    Raises:
        ValueError: If ``market_price`` is not finite, if a solver option is
            out of range (``tol`` and the bracket edges must be positive and
            ``vol_upper`` must exceed ``vol_lower``), if ``max_iter`` is not
            positive, or if ``market_price`` lies outside the no-arbitrage
            bounds.

    Solver options are validated before any pricing runs, so a misconfigured
    call fails immediately rather than returning a plausible wrong number:

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> price = black_scholes_price(call)
    >>> implied_volatility(price, call, tol=0.0)
    Traceback (most recent call last):
        ...
    ValueError: tol must be positive, got 0.0
    >>> implied_volatility(price, call, max_iter=0)
    Traceback (most recent call last):
        ...
    ValueError: max_iter must be positive, got 0
    >>> implied_volatility(price, call, vol_lower=0.0)
    Traceback (most recent call last):
        ...
    ValueError: vol_lower must be positive, got 0.0
    >>> implied_volatility(price, call, vol_upper=0.1, vol_lower=0.2)
    Traceback (most recent call last):
        ...
    ValueError: vol_upper (0.1) must be greater than vol_lower (0.2)
    >>> implied_volatility(float("nan"), call)
    Traceback (most recent call last):
        ...
    ValueError: market_price must be a finite number, got nan

    Round-tripping a known price recovers the input volatility:

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> price = black_scholes_price(call)
    >>> round(implied_volatility(price, call), 8)
    0.2

    >>> put = OptionParams(100.0, 110.0, 0.5, 0.03, 0.35, OptionType.PUT)
    >>> price = black_scholes_price(put)
    >>> round(implied_volatility(price, put), 8)
    0.35

    Works with a dividend yield, and across strikes:

    >>> for K in (80.0, 100.0, 120.0):
    ...     p = OptionParams(100.0, K, 1.0, 0.04, 0.28, OptionType.CALL, 0.015)
    ...     round(implied_volatility(black_scholes_price(p), p), 8)
    0.28
    0.28
    0.28

    Prices outside the arbitrage bounds are rejected rather than silently
    extrapolated:

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> implied_volatility(0.01, call)
    Traceback (most recent call last):
        ...
    ValueError: market price 0.010000 outside no-arbitrage bounds [4.877058, 100.000000]
    >>> implied_volatility(500.0, call)
    Traceback (most recent call last):
        ...
    ValueError: market price 500.000000 outside no-arbitrage bounds [4.877058, 100.000000]

    Put bounds are checked too:

    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> implied_volatility(200.0, put)
    Traceback (most recent call last):
        ...
    ValueError: market price 200.000000 outside no-arbitrage bounds [0.000000, 95.122942]
    """
    # Validate the solver configuration first. A bad bracket or tolerance would
    # otherwise produce a plausible-looking number instead of an error. Also,
    # non-finite inputs compare false against every bound, so market_price must
    # be checked explicitly rather than relying on the no-arbitrage test below,
    # which a NaN or infinite price would slip straight past.
    if not math.isfinite(market_price):
        raise ValueError(f"market_price must be a finite number, got {market_price}")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if not math.isfinite(max_iter) or max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if not math.isfinite(vol_lower) or vol_lower <= 0:
        raise ValueError(f"vol_lower must be positive, got {vol_lower}")
    if not math.isfinite(vol_upper) or vol_upper <= vol_lower:
        raise ValueError(f"vol_upper ({vol_upper}) must be greater than vol_lower ({vol_lower})")

    lower, upper = arbitrage_bounds(params)
    tolerance = 1e-8
    if not (lower - tolerance <= market_price <= upper + tolerance):
        raise ValueError(
            f"market price {market_price:.6f} outside no-arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}]"
        )

    def price_at(vol: float) -> float:
        return black_scholes_price(params.replace_volatility(vol))

    f_low = price_at(vol_lower) - market_price
    f_high = price_at(vol_upper) - market_price

    if f_low > 0 or f_high < 0:
        raise ValueError(
            f"market price {market_price:.6f} not reachable with volatility in "
            f"[{vol_lower}, {vol_upper}]; widen vol_lower/vol_upper"
        )

    # Newton warm start. Only adopt the narrowed bracket if it still straddles
    # the root, otherwise keep the original full bracket. Deep out-of-the-money
    # options have a nearly flat price in vol, and a poor Newton guess would
    # otherwise collapse the bracket away from the true answer.
    guess = _newton_iv(market_price, params, tol=tol)
    if vol_lower < guess < vol_upper:
        narrowed_low = max(vol_lower, guess - 0.05)
        narrowed_high = min(vol_upper, guess + 0.05)
        nf_low = price_at(narrowed_low) - market_price
        nf_high = price_at(narrowed_high) - market_price
        if nf_low <= 0 <= nf_high:
            vol_lower, vol_upper = narrowed_low, narrowed_high
            f_low, f_high = nf_low, nf_high

    for _ in range(max_iter):
        mid = 0.5 * (vol_lower + vol_upper)
        f_mid = price_at(mid) - market_price
        if abs(f_mid) < tol or (vol_upper - vol_lower) < tol:
            return float(mid)
        if f_low * f_mid <= 0:
            vol_upper, f_high = mid, f_mid
        else:
            vol_lower, f_low = mid, f_mid

    return float(0.5 * (vol_lower + vol_upper))


def _newton_iv(
    market_price: float,
    params: OptionParams,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility, used only as a bisection warm start.

    The iteration is guarded against a zero or near-zero Vega, which happens
    for deep in-the-money options at very short maturities.

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> round(_newton_iv(black_scholes_price(call), call), 6)
    0.2
    """
    T, q = params.time_to_maturity, params.dividend_yield
    disc_q = math.exp(-q * T)

    # Brenner-Subrahmanyam initial guess.
    vol = math.sqrt(2 * math.pi / T) * market_price / (params.spot * disc_q)
    vol = min(max(vol, 1e-6), 5.0)

    for _ in range(max_iter):
        probe = params.replace_volatility(vol)
        diff = black_scholes_price(probe) - market_price
        if abs(diff) < tol:
            return float(vol)
        vega_unit = black_scholes_greeks(probe).vega / PERCENT
        if vega_unit < 1e-12:
            break
        vol = min(max(vol - diff / vega_unit, 1e-6), 5.0)

    return float(vol)


def monte_carlo_price(
    params: OptionParams,
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int | None = None,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Simulate European option payoff under geometric Brownian motion.

    This is a reference implementation used to cross-check the closed form.
    It is fully vectorized, and the reported standard error accounts for
    antithetic pairing (the error is measured on pair averages, which are
    the independent samples).

    Args:
        params: Option parameters.
        n_paths: Total number of paths. Rounded down to an even number when
            ``antithetic`` is True.
        n_steps: Time steps used to build each path. One step is exact for
            European options; more steps only adds sampling noise.
        seed: Optional seed for reproducibility.
        antithetic: Whether to use antithetic variates to cut variance.

    Returns:
        ``(price, standard_error)`` as native floats.

    Raises:
        ValueError: If ``n_paths`` or ``n_steps`` is not positive, or if
            ``n_paths`` is too small to pair.

    The simulation converges to the analytical price:

    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    >>> mc, se = monte_carlo_price(call, n_paths=200_000, seed=42)
    >>> abs(mc - black_scholes_price(call)) < 3 * se
    True

    Seeded runs are exactly reproducible:

    >>> a, _ = monte_carlo_price(call, n_paths=20_000, seed=7)
    >>> b, _ = monte_carlo_price(call, n_paths=20_000, seed=7)
    >>> a == b
    True

    Antithetic variates cut the standard error at equal path count:

    >>> _, se_anti = monte_carlo_price(call, n_paths=50_000, seed=3, antithetic=True)
    >>> _, se_plain = monte_carlo_price(call, n_paths=50_000, seed=3, antithetic=False)
    >>> se_anti < se_plain
    True

    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> mc, se = monte_carlo_price(put, n_paths=200_000, seed=11)
    >>> abs(mc - black_scholes_price(put)) < 3 * se
    True

    >>> monte_carlo_price(call, n_paths=0)
    Traceback (most recent call last):
        ...
    ValueError: n_paths must be positive, got 0
    >>> monte_carlo_price(call, n_paths=4, n_steps=0)
    Traceback (most recent call last):
        ...
    ValueError: n_steps must be positive, got 0
    >>> monte_carlo_price(call, n_paths=2, antithetic=True)
    Traceback (most recent call last):
        ...
    ValueError: n_paths must be at least 4 when antithetic=True, got 2
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    if antithetic and n_paths < 4:
        raise ValueError(f"n_paths must be at least 4 when antithetic=True, got {n_paths}")

    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    vol, q = params.volatility, params.dividend_yield

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    drift = (r - q - 0.5 * vol * vol) * dt
    vol_step = vol * math.sqrt(dt)
    disc = math.exp(-r * T)

    is_call = params.option_type is OptionType.CALL

    def payoff_of(terminal: np.ndarray) -> np.ndarray:
        return np.maximum(terminal - K, 0.0) if is_call else np.maximum(K - terminal, 0.0)

    if antithetic:
        n_pairs = n_paths // 2
        z = rng.standard_normal((n_pairs, n_steps))
        # Antithetic partner is the same path with every shock negated.
        up = S * np.exp(np.sum(drift + vol_step * z, axis=1))
        down = S * np.exp(np.sum(drift - vol_step * z, axis=1))
        samples = 0.5 * (payoff_of(up) + payoff_of(down))
        price = disc * float(samples.mean())
        std_err = disc * float(samples.std(ddof=1)) / math.sqrt(n_pairs)
    else:
        z = rng.standard_normal((n_paths, n_steps))
        terminal = S * np.exp(np.sum(drift + vol_step * z, axis=1))
        samples = payoff_of(terminal)
        price = disc * float(samples.mean())
        std_err = disc * float(samples.std(ddof=1)) / math.sqrt(n_paths)

    return float(price), float(std_err)


__all__ = [
    "DAYS_PER_YEAR",
    "PERCENT",
    "Greeks",
    "OptionParams",
    "OptionType",
    "PricingResult",
    "__version__",
    "arbitrage_bounds",
    "black_scholes_greeks",
    "black_scholes_price",
    "implied_volatility",
    "monte_carlo_price",
    "price_option",
]
