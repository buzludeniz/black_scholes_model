"""
American option pricing.

A European option can only be exercised at expiry. An American option can be
exercised at any time up to expiry, so it is worth at least as much as the
European option and at least its intrinsic value. That early-exercise right has
no closed form under Black-Scholes dynamics, so this module provides two
standard methods:

- :func:`binomial_price`: the Cox-Ross-Rubinstein (1979) binomial tree, which
  converges to the true value as the number of steps grows and is the reference
  method here.
- :func:`barone_adesi_whaley_price`: the Barone-Adesi and Whaley (1987)
  quadratic approximation, a closed-form-plus-one-root-find method that is far
  faster and accurate to a few cents for typical inputs.

Both use the same model as the rest of the package: geometric Brownian motion
with risk-free rate ``r``, volatility ``sigma`` and continuous dividend yield
``q``, so the cost of carry is ``b = r - q``.

Two classical results bound the answers and are tested:

- With ``q = 0`` (more generally ``q <= 0``), early exercise of a call is never
  optimal (Merton 1973), so the American call equals the European call.
- With ``r = 0`` early exercise of a put is never optimal, so the American put
  equals the European put.

References
----------
J. C. Cox, S. A. Ross and M. Rubinstein, *Option pricing: a simplified
approach*, Journal of Financial Economics 7 (1979) 229-263.

G. Barone-Adesi and R. E. Whaley, *Efficient analytic approximation of American
option values*, Journal of Finance 42 (1987) 301-320.

E. G. Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
McGraw-Hill (2007), the Barone-Adesi and Whaley section.

>>> from black_scholes import OptionParams, OptionType
>>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
>>> round(binomial_price(put, steps=2000), 3)
6.09
>>> round(barone_adesi_whaley_price(put), 3)
6.098
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from black_scholes import OptionParams, OptionType, black_scholes_price

__all__ = [
    "DEFAULT_STEPS",
    "american_price",
    "barone_adesi_whaley_price",
    "binomial_price",
]

#: Default number of tree steps. The CRR price oscillates as it converges, with
#: an error roughly proportional to ``1/steps``; 1000 steps puts a typical
#: at-the-money price within about a tenth of a cent of its limit.
DEFAULT_STEPS = 1000


def binomial_price(
    params: OptionParams, steps: int = DEFAULT_STEPS, *, american: bool = True
) -> float:
    """Price an option on a Cox-Ross-Rubinstein binomial tree.

    The tree has ``n = steps`` periods of length ``dt = T/n``. Over one period
    the underlying moves up by ``u`` or down by ``d`` with risk-neutral
    probability ``p``::

        u = exp(sigma * sqrt(dt)),    d = 1/u
        p = (exp((r - q) * dt) - d) / (u - d)

    Node ``(i, j)`` (time step ``i``, ``j`` up-moves) has underlying price
    ``S * u^j * d^(i-j)``. Option values are rolled back from the payoff at
    expiry with::

        V(i, j) = exp(-r*dt) * (p * V(i+1, j+1) + (1 - p) * V(i+1, j))

    and, for an American option, replaced at every node by the larger of that
    continuation value and the immediate exercise value ``max(S - K, 0)`` for a
    call or ``max(K - S, 0)`` for a put.

    Args:
        params: Option parameters, including the dividend yield ``q``.
        steps: Number of time steps. More steps converge towards the
            continuous-time value; the error shrinks roughly as ``1/steps``
            with a characteristic odd/even oscillation.
        american: ``True`` (default) allows exercise at every node; ``False``
            prices the European option on the same tree, which is useful as a
            control variate.

    Returns:
        The option value as a native ``float``.

    Raises:
        ValueError: If ``steps`` is not a positive integer, or if the tree is
            too coarse for the carry, so that ``p`` falls outside ``(0, 1)``.
            That happens when ``|r - q| * sqrt(dt) >= sigma`` roughly, and more
            steps fix it.

    Merton's result: with no dividend an American call is never exercised
    early, so the tree gives the European tree value exactly:

    >>> from black_scholes import OptionParams, OptionType
    >>> call = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20)
    >>> binomial_price(call, 500) == binomial_price(call, 500, american=False)
    True

    Hull's worked example (*Options, Futures, and Other Derivatives*): a
    5-month American put, S = K = 50, r = 10%, sigma = 40%, on a 5-step tree:

    >>> hull = OptionParams(50.0, 50.0, 5 / 12, 0.10, 0.40, OptionType.PUT)
    >>> round(binomial_price(hull, 5), 2)
    4.49

    >>> binomial_price(call, 0)
    Traceback (most recent call last):
        ...
    ValueError: steps must be a positive integer, got 0
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError(f"steps must be a positive integer, got {steps}")
    n = int(steps)

    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    sigma, q = params.volatility, params.dividend_yield
    is_call = params.option_type is OptionType.CALL

    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.6f} is outside (0, 1) with {n} steps; "
            "the tree is too coarse for this carry, so increase steps"
        )
    disc = math.exp(-r * dt)
    disc_up, disc_down = disc * p, disc * (1.0 - p)

    # Underlying at level i with j = 0..i up-moves is S * u^(2j - i). Each
    # level is computed from S directly rather than by repeatedly multiplying
    # the previous level by u: accumulated rounding would otherwise leave the
    # root spot a few ulps off S, and an option exercised there would come out
    # a hair below its intrinsic value.
    log_step = sigma * math.sqrt(dt)
    j = np.arange(n + 1, dtype=float)
    spots = S * np.exp(log_step * (2.0 * j - n))
    values = np.maximum(spots - K, 0.0) if is_call else np.maximum(K - spots, 0.0)

    for i in range(n - 1, -1, -1):
        values = disc_up * values[1:] + disc_down * values[:-1]
        if american:
            spots = S * np.exp(log_step * (2.0 * j[: i + 1] - i))
            exercise = spots - K if is_call else K - spots
            np.maximum(values, exercise, out=values)

    return float(values[0])


def _critical_price(params: OptionParams, q_exp: float) -> float:
    """Solve the Barone-Adesi-Whaley critical-price equation.

    For a call, ``S*`` solves::

        S* - K = c(S*) + (1 - exp((b - r)T) N(d1(S*))) * S* / q2

    and for a put, ``S**`` solves::

        K - S** = p(S**) - (1 - exp((b - r)T) N(-d1(S**))) * S** / q1

    where ``c`` and ``p`` are the European prices. Each side is a smooth
    function of the spot with exactly one crossing on the relevant half-line
    (above the strike for a call, below it for a put), so a bracketing solver
    is used rather than the Newton iteration of the paper: it cannot diverge.
    """
    K, T = params.strike, params.time_to_maturity
    r, q, sigma = params.risk_free_rate, params.dividend_yield, params.volatility
    b = r - q
    carry_disc = math.exp((b - r) * T)
    sqrt_t = math.sqrt(T)
    is_call = params.option_type is OptionType.CALL

    def gap(spot: float) -> float:
        probe = OptionParams(spot, K, T, r, sigma, params.option_type, q)
        euro = black_scholes_price(probe)
        d1 = (math.log(spot / K) + (b + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        if is_call:
            return euro + (1.0 - carry_disc * float(norm.cdf(d1))) * spot / q_exp - (spot - K)
        return euro - (1.0 - carry_disc * float(norm.cdf(-d1))) * spot / q_exp - (K - spot)

    if is_call:
        # gap(K) > 0, and gap falls linearly with slope (1 - e^((b-r)T))(1/q2 - 1)
        # < 0 for large spots, so doubling the upper edge brackets the root.
        low, high = K, 2.0 * K
        while gap(high) > 0.0:
            low, high = high, 2.0 * high
    else:
        # gap(K) > 0 and gap -> K*exp(-rT) - K < 0 as the spot falls to zero.
        high, low = K, 0.5 * K
        while gap(low) > 0.0:
            high, low = low, 0.5 * low
    return float(brentq(gap, low, high, xtol=1e-12 * K, rtol=1e-14, maxiter=200))


def barone_adesi_whaley_price(params: OptionParams) -> float:
    """Price an American option with the Barone-Adesi and Whaley approximation.

    The early-exercise premium satisfies the Black-Scholes PDE. Barone-Adesi and
    Whaley (1987) write the American price as European price plus premium,
    drop one small term from the PDE for that premium, and solve what remains
    in closed form. With cost of carry ``b = r - q``::

        M = 2r / sigma^2,   N = 2b / sigma^2,   h = 1 - exp(-rT)

        q2 = (-(N - 1) + sqrt((N - 1)^2 + 4M/h)) / 2      (call, q2 > 1)
        q1 = (-(N - 1) - sqrt((N - 1)^2 + 4M/h)) / 2      (put,  q1 < 0)

    **Call.** With ``S*`` the critical spot above which exercise is optimal::

        A2 = (S*/q2) * (1 - exp((b - r)T) N(d1(S*)))
        C  = c(S) + A2 (S/S*)^q2     if S < S*
        C  = S - K                   if S >= S*

    **Put.** With ``S**`` the critical spot below which exercise is optimal::

        A1 = -(S**/q1) * (1 - exp((b - r)T) N(-d1(S**)))
        P  = p(S) + A1 (S/S**)^q1    if S > S**
        P  = K - S                   if S <= S**

    ``c`` and ``p`` are the European Black-Scholes prices and
    ``d1(x) = (ln(x/K) + (b + sigma^2/2)T) / (sigma sqrt(T))``. The critical
    spots solve the smooth-pasting equations in :func:`_critical_price`.

    Two exact cases bypass the approximation:

    - a call with ``q <= 0`` (``b >= r``) is never exercised early, so its
      value is the European price;
    - a put with ``r = 0`` is never exercised early either, so its value is the
      European price. (As ``r -> 0`` the formulas above use the limit
      ``M/h -> 2/(sigma^2 T)``, which is applied for a call with ``r = 0``.)

    Returns:
        The option value as a native ``float``.

    Haug's worked example (an American put on a future, so ``b = 0``, which is
    ``q = r`` here): S = K = 100, T = 0.5, r = 10%, sigma = 25%, published
    value 6.801362:

    >>> from black_scholes import OptionParams, OptionType
    >>> put = OptionParams(100.0, 100.0, 0.5, 0.10, 0.25, OptionType.PUT, 0.10)
    >>> round(barone_adesi_whaley_price(put), 4)
    6.8013

    A deep in-the-money put beyond the critical spot is worth its intrinsic
    value:

    >>> deep = OptionParams(50.0, 100.0, 1.0, 0.08, 0.20, OptionType.PUT)
    >>> barone_adesi_whaley_price(deep)
    50.0
    """
    S, K = params.spot, params.strike
    T, r = params.time_to_maturity, params.risk_free_rate
    sigma, q = params.volatility, params.dividend_yield
    b = r - q
    is_call = params.option_type is OptionType.CALL
    european = black_scholes_price(params)

    if is_call and b >= r:
        return european
    if not is_call and r == 0.0:
        return european

    sigma2 = sigma * sigma
    big_n = 2.0 * b / sigma2
    # 4M/h with M = 2r/sigma^2 and h = 1 - exp(-rT); continuous at r = 0.
    m_over_h = 2.0 / (sigma2 * T) if r == 0.0 else (2.0 * r / sigma2) / -math.expm1(-r * T)
    root = math.sqrt((big_n - 1.0) ** 2 + 4.0 * m_over_h)
    q_exp = 0.5 * (-(big_n - 1.0) + root) if is_call else 0.5 * (-(big_n - 1.0) - root)

    critical = _critical_price(params, q_exp)
    carry_disc = math.exp((b - r) * T)
    d1 = (math.log(critical / K) + (b + 0.5 * sigma2) * T) / (sigma * math.sqrt(T))

    if is_call:
        if critical <= S:
            return float(S - K)
        a2 = (critical / q_exp) * (1.0 - carry_disc * float(norm.cdf(d1)))
        return float(european + a2 * (S / critical) ** q_exp)

    if critical >= S:
        return float(K - S)
    a1 = -(critical / q_exp) * (1.0 - carry_disc * float(norm.cdf(-d1)))
    return float(european + a1 * (S / critical) ** q_exp)


def american_price(
    params: OptionParams,
    *,
    method: Literal["crr", "baw"] = "crr",
    steps: int = DEFAULT_STEPS,
) -> float:
    """Price an American option with the chosen method.

    Args:
        params: Option parameters.
        method: ``"crr"`` (default) for the binomial tree, ``"baw"`` for the
            Barone-Adesi-Whaley approximation.
        steps: Tree steps; ignored by ``"baw"``.

    Raises:
        ValueError: For an unknown ``method``, or as raised by the chosen
            method.

    >>> from black_scholes import OptionParams, OptionType
    >>> put = OptionParams(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
    >>> abs(american_price(put) - american_price(put, method="baw")) < 0.01
    True
    >>> american_price(put, method="lsm")
    Traceback (most recent call last):
        ...
    ValueError: method must be 'crr' or 'baw', got 'lsm'
    """
    if method == "crr":
        return binomial_price(params, steps)
    if method == "baw":
        return barone_adesi_whaley_price(params)
    raise ValueError(f"method must be 'crr' or 'baw', got {method!r}")
