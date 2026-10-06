"""
Worked examples for the black-scholes package.

Run with:  python examples/basic_usage.py

Output is ASCII only so it renders correctly in the Windows console.
"""

from __future__ import annotations

import math

from black_scholes import (
    OptionParams,
    OptionType,
    arbitrage_bounds,
    black_scholes_price,
    implied_volatility,
    monte_carlo_price,
    price_option,
)


def rule(title: str) -> None:
    print("=" * 62)
    print(title)
    print("=" * 62)


def example_basic_pricing() -> None:
    """Price an at-the-money call and show its Greeks."""
    rule("Example 1: ATM Call Pricing")

    params = OptionParams(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type=OptionType.CALL,
    )
    result = price_option(params)

    print(f"Spot      : ${params.spot:.2f}")
    print(f"Strike    : ${params.strike:.2f}")
    print(f"Maturity  : {params.time_to_maturity:.2f} years")
    print(f"Rate      : {params.risk_free_rate:.1%}")
    print(f"Volatility: {params.volatility:.1%}")
    print()
    print(result)
    print()


def example_put_call_parity() -> None:
    """Show that C - P equals the discounted forward spread exactly."""
    rule("Example 2: Put-Call Parity")

    S, K, T, r, vol = 100.0, 105.0, 0.5, 0.04, 0.22

    call = black_scholes_price(OptionParams(S, K, T, r, vol, OptionType.CALL))
    put = black_scholes_price(OptionParams(S, K, T, r, vol, OptionType.PUT))

    parity = S - K * math.exp(-r * T)
    diff = call - put

    print(f"{'Call price':22} {call:>16.10f}")
    print(f"{'Put price':22} {put:>16.10f}")
    print(f"{'C - P':22} {diff:>16.10f}")
    print(f"{'S - K*e^(-rT)':22} {parity:>16.10f}")
    print(f"{'Residual':22} {abs(diff - parity):>16.2e}")
    print("\nZero residual means the identity holds to machine precision.")
    print()


def example_implied_vol() -> None:
    """Invert known prices back to the volatility that produced them."""
    rule("Example 3: Implied Volatility Round-Trip")

    S, T, r, true_vol = 100.0, 1.0, 0.05, 0.20

    print(f"{'Strike':>8} {'Model Price':>13} {'Implied Vol':>13}  Position")
    print("-" * 54)

    for K in (90.0, 100.0, 110.0):
        params = OptionParams(S, K, T, r, true_vol, OptionType.CALL)
        price = price_option(params).price
        iv = implied_volatility(price, params)
        position = "ITM" if K < S else ("ATM" if K == S else "OTM")
        print(f"{K:>8.2f} {price:>13.6f} {iv:>12.4%}  {position}")

    print()


def example_volatility_smile() -> None:
    """A smile is just a different implied vol at each strike."""
    rule("Example 4: Reading a Volatility Smile")

    S, T, r = 100.0, 0.5, 0.03
    strikes = (80.0, 90.0, 100.0, 110.0, 120.0)

    # Prices quoted at a deliberately non-flat volatility surface.
    market_prices = {80.0: 21.30, 90.0: 13.05, 100.0: 7.20, 110.0: 3.55, 120.0: 1.62}

    print(f"{'Strike':>8} {'Mkt Price':>11} {'Implied Vol':>13}")
    print("-" * 36)

    for K in strikes:
        params = OptionParams(S, K, T, r, 0.20, OptionType.CALL)
        lo, hi = arbitrage_bounds(params)
        price = market_prices[K]
        if not (lo <= price <= hi):
            print(f"{K:>8.2f} {price:>11.4f} {'(no-arbitrage)':>13}")
            continue
        iv = implied_volatility(price, params)
        print(f"{K:>8.2f} {price:>11.4f} {iv:>12.2%}")

    print("\nNote: a smile means deep-in and deep-out strikes carry higher vol.")
    print()


def example_monte_carlo() -> None:
    """Show the simulation converging on the closed form."""
    rule("Example 5: Monte Carlo Validation")

    params = OptionParams(100.0, 105.0, 0.5, 0.04, 0.25, OptionType.CALL)
    analytic = price_option(params).price

    print(f"Analytical price: {analytic:.6f}\n")
    print(f"{'Paths':>10} {'MC Price':>12} {'Std Error':>11} {'Abs Diff':>10}  Verdict")
    print("-" * 60)

    for n_paths in (10_000, 100_000, 500_000, 2_000_000):
        mc_price, mc_se = monte_carlo_price(params, n_paths=n_paths, seed=42)
        diff = abs(mc_price - analytic)
        ok = diff < 3 * mc_se
        print(
            f"{n_paths:>10,} {mc_price:>12.6f} {mc_se:>11.6f} {diff:>10.6f}"
            f"  {'in tolerance' if ok else 'OUT OF TOLERANCE'}"
        )

    print("\nError falls as 1/sqrt(paths): 100x the paths gives 10x the precision.")
    print()


def example_greeks_across_strikes() -> None:
    """Risk profiles change shape with moneyness."""
    rule("Example 6: Greeks Across Strikes")

    S, T, r, vol = 100.0, 0.25, 0.05, 0.30

    print(f"{'Strike':>7} {'Delta':>8} {'Gamma':>9} {'Vega':>8} {'Theta':>9} {'Rho':>8}")
    print("-" * 54)

    for K in (80.0, 90.0, 100.0, 110.0, 120.0):
        params = OptionParams(S, K, T, r, vol, OptionType.CALL)
        g = price_option(params).greeks
        print(
            f"{K:>7.0f} {g.delta:>8.4f} {g.gamma:>9.6f} "
            f"{g.vega:>8.4f} {g.theta:>9.4f} {g.rho:>8.4f}"
        )

    print("\nDelta is steepest and Gamma peaks at the money.")
    print()


def example_dividend_effect() -> None:
    """A dividend lowers calls and raises puts, but not by the same amount."""
    rule("Example 7: Dividend Yield Effect")

    T = 1.0

    def both(q: float) -> tuple[float, float]:
        call = price_option(OptionParams(100.0, 100.0, T, 0.05, 0.20, OptionType.CALL, q)).price
        put = price_option(OptionParams(100.0, 100.0, T, 0.05, 0.20, OptionType.PUT, q)).price
        return call, put

    call_base, put_base = both(0.0)
    call_div, put_div = both(0.03)

    print(f"{'':22} {'q = 0%':>12} {'q = 3%':>12} {'change':>12}")
    print("-" * 60)
    print(f"{'Call price':22} {call_base:>12.4f} {call_div:>12.4f} {call_div - call_base:>12.4f}")
    print(f"{'Put price':22} {put_base:>12.4f} {put_div:>12.4f} {put_div - put_base:>12.4f}")
    print()

    drop_call = call_base - call_div
    rise_put = put_div - put_base
    forward_loss = 100.0 * (1.0 - math.exp(-0.03 * T))

    print(f"Call falls by        {drop_call:.4f}")
    print(f"Put rises by         {rise_put:.4f}")
    print(f"Sum of the two moves {drop_call + rise_put:.4f}")
    print(f"Discounted stock loss {forward_loss:.4f}")
    print()
    print("The call's loss and the put's gain sum to the drop in the stock's")
    print("discounted value. That is put-call parity (C - P = S*e^(-qT) - K*e^(-rT))")
    print("at work: a dividend is not a symmetric transfer between the two sides.")
    print()


def example_arbitrage_bounds() -> None:
    """Every model price must sit inside the no-arbitrage interval."""
    rule("Example 8: No-Arbitrage Bounds")

    S, K, T, r, vol = 120.0, 100.0, 0.75, 0.04, 0.35

    print(f"{'Type':>6} {'Lower':>12} {'Model':>12} {'Upper':>12}  Inside")
    print("-" * 62)

    for option_type in (OptionType.CALL, OptionType.PUT):
        params = OptionParams(S, K, T, r, vol, option_type)
        lo, hi = arbitrage_bounds(params)
        price = price_option(params).price
        label = option_type.value.capitalize()
        print(
            f"{label:>6} {lo:>12.4f} {price:>12.4f} {hi:>12.4f}"
            f"  {'yes' if lo <= price <= hi else 'NO'}"
        )

    print("\nA price outside these bounds would be a free lunch.")
    print()


EXAMPLES = [
    example_basic_pricing,
    example_put_call_parity,
    example_implied_vol,
    example_volatility_smile,
    example_monte_carlo,
    example_greeks_across_strikes,
    example_dividend_effect,
    example_arbitrage_bounds,
]


def main() -> None:
    for example in EXAMPLES:
        example()


if __name__ == "__main__":
    main()
