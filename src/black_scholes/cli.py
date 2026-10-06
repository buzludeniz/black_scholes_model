"""
Command-line interface for Black-Scholes pricer.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from black_scholes import (
    OptionParams,
    OptionType,
    implied_volatility,
    monte_carlo_price,
    price_option,
)
from black_scholes.surface import SVIError, VolSurface

app = typer.Typer(
    name="bs",
    help="Black-Scholes European Option Pricer",
    add_completion=False,
    no_args_is_help=True,
)
# Force ASCII-safe output on Windows
console = Console(legacy_windows=False, safe_box=True)


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


def _parse_option_type(value: str) -> OptionType:
    v = value.lower()
    if v in ("c", "call"):
        return OptionType.CALL
    if v in ("p", "put"):
        return OptionType.PUT
    raise typer.BadParameter("Option type must be 'call'/'c' or 'put'/'p'")


@app.command()
def price(
    spot: Annotated[float, typer.Argument(help="Current underlying price (S)")],
    strike: Annotated[float, typer.Argument(help="Strike price (K)")],
    time: Annotated[float, typer.Argument(help="Time to maturity in years (T)")],
    rate: Annotated[float, typer.Argument(help="Risk-free rate (r), e.g. 0.05 for 5%")],
    vol: Annotated[float, typer.Argument(help="Volatility (vol), e.g. 0.2 for 20%")],
    option_type: Annotated[
        str, typer.Argument(help="Option type: call/c or put/p", parser=_parse_option_type)
    ],
    div_yield: Annotated[
        float, typer.Option("--div-yield", "-q", help="Continuous dividend yield")
    ] = 0.0,
    format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TEXT,
    mc: Annotated[bool, typer.Option("--mc", help="Run Monte Carlo validation")] = False,
    mc_paths: Annotated[int, typer.Option("--mc-paths", help="Monte Carlo paths")] = 100_000,
) -> None:
    """Price a European option with full Greeks."""
    opt = cast(OptionType, option_type)
    params = OptionParams(
        spot=spot,
        strike=strike,
        time_to_maturity=time,
        risk_free_rate=rate,
        volatility=vol,
        option_type=opt,
        dividend_yield=div_yield,
    )

    result = price_option(params)

    if format == OutputFormat.JSON:
        output = {
            "price": result.price,
            "greeks": result.greeks.as_dict(),
            "params": {
                "spot": params.spot,
                "strike": params.strike,
                "time_to_maturity": params.time_to_maturity,
                "risk_free_rate": params.risk_free_rate,
                "volatility": params.volatility,
                "option_type": params.option_type.value,
                "dividend_yield": params.dividend_yield,
            },
        }
        if mc:
            mc_price, mc_se = monte_carlo_price(params, n_paths=mc_paths)
            output["monte_carlo"] = {
                "price": mc_price,
                "std_error": mc_se,
                "diff": result.price - mc_price,
                "within_2se": abs(result.price - mc_price) <= 2 * mc_se,
            }
        console.print_json(json.dumps(output))
        return

    # Rich text output
    table = Table(title=f"Black-Scholes {params.option_type.value.capitalize()} Option")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Spot (S)", f"{params.spot:.4f}")
    table.add_row("Strike (K)", f"{params.strike:.4f}")
    table.add_row("Time (T)", f"{params.time_to_maturity:.4f} years")
    table.add_row("Rate (r)", f"{params.risk_free_rate:.4%}")
    table.add_row("Vol (vol)", f"{params.volatility:.4%}")
    table.add_row("Div Yield (q)", f"{params.dividend_yield:.4%}")
    table.add_row("", "")
    table.add_row("Price", f"[bold]{result.price:.6f}[/bold]")
    table.add_row("Delta", f"{result.greeks.delta:.6f}")
    table.add_row("Gamma", f"{result.greeks.gamma:.6f}")
    table.add_row("Vega (per 1%)", f"{result.greeks.vega:.6f}")
    table.add_row("Theta (per day)", f"{result.greeks.theta:.6f}")
    table.add_row("Rho (per 1%)", f"{result.greeks.rho:.6f}")

    console.print(table)

    if mc:
        mc_price, mc_se = monte_carlo_price(params, n_paths=mc_paths)
        diff = result.price - mc_price
        within = "OK" if abs(diff) <= 2 * mc_se else "FAIL"

        mc_table = Table(title="Monte Carlo Validation")
        mc_table.add_column("Metric", style="cyan")
        mc_table.add_column("Value", style="green")
        mc_table.add_row("MC Price", f"{mc_price:.6f}")
        mc_table.add_row("Std Error", f"{mc_se:.6f}")
        mc_table.add_row("Difference", f"{diff:.6f}")
        mc_table.add_row("Within 2 SE", within)
        console.print(mc_table)


@app.command()
def iv(
    market_price: Annotated[float, typer.Argument(help="Observed market price")],
    spot: Annotated[float, typer.Argument(help="Current underlying price (S)")],
    strike: Annotated[float, typer.Argument(help="Strike price (K)")],
    time: Annotated[float, typer.Argument(help="Time to maturity in years (T)")],
    rate: Annotated[float, typer.Argument(help="Risk-free rate (r)")],
    option_type: Annotated[
        str, typer.Argument(help="Option type: call/c or put/p", parser=_parse_option_type)
    ],
    div_yield: Annotated[
        float, typer.Option("--div-yield", "-q", help="Continuous dividend yield")
    ] = 0.0,
    format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TEXT,
) -> None:
    """Calculate implied volatility from market price."""
    params = OptionParams(
        spot=spot,
        strike=strike,
        time_to_maturity=time,
        risk_free_rate=rate,
        # implied_volatility() solves for this value, so it is only a seed.
        volatility=0.2,
        option_type=cast(OptionType, option_type),
        dividend_yield=div_yield,
    )

    try:
        iv = implied_volatility(market_price, params)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    if format == OutputFormat.JSON:
        console.print_json(json.dumps({"implied_volatility": iv, "implied_vol_pct": iv * 100}))
        return

    panel = Panel.fit(
        f"[bold]Implied Volatility:[/bold] {iv:.6f} ({iv * 100:.2f}%)\n"
        f"Market Price: {market_price:.6f}\n"
        f"Spot: {spot:.4f} | Strike: {strike:.4f} | Time: {time:.4f} | Rate: {rate:.4%}",
        title="Implied Volatility",
        border_style="green",
    )
    console.print(panel)


@app.command()
def surface(
    spot: Annotated[float, typer.Option("--spot", "-S", help="Current underlying price")],
    time: Annotated[float, typer.Option("--time", "-T", help="Time to maturity in years")],
    rate: Annotated[float, typer.Option("--rate", "-r", help="Risk-free rate")],
    vol: Annotated[float, typer.Option("--vol", "-v", help="Volatility")],
    strikes: Annotated[
        str, typer.Option("--strikes", "-K", help="Comma-separated strikes")
    ] = "80,90,100,110,120",
    div_yield: Annotated[float, typer.Option("--div-yield", "-q", help="Dividend yield")] = 0.0,
) -> None:
    """Generate a volatility smile / price surface across strikes."""
    strike_list = [float(k.strip()) for k in strikes.split(",")]

    table = Table(title=f"Option Surface (S={spot}, T={time}, r={rate:.2%}, vol={vol:.2%})")
    table.add_column("Strike", justify="right")
    table.add_column("Call Price", justify="right")
    table.add_column("Call Delta", justify="right")
    table.add_column("Put Price", justify="right")
    table.add_column("Put Delta", justify="right")
    table.add_column("IV (Call)", justify="right")
    table.add_column("IV (Put)", justify="right")

    for K in strike_list:
        call_params = OptionParams(spot, K, time, rate, vol, OptionType.CALL, div_yield)
        put_params = OptionParams(spot, K, time, rate, vol, OptionType.PUT, div_yield)

        call_res = price_option(call_params)
        put_res = price_option(put_params)

        # IV from own price (should equal input vol)
        call_iv = implied_volatility(call_res.price, call_params)
        put_iv = implied_volatility(put_res.price, put_params)

        table.add_row(
            f"{K:.2f}",
            f"{call_res.price:.4f}",
            f"{call_res.greeks.delta:.4f}",
            f"{put_res.price:.4f}",
            f"{put_res.greeks.delta:.4f}",
            f"{call_iv:.4%}",
            f"{put_iv:.4%}",
        )

    console.print(table)


@app.command()
def greeks(
    spot: Annotated[float, typer.Argument(help="Current underlying price (S)")],
    strike: Annotated[float, typer.Argument(help="Strike price (K)")],
    time: Annotated[float, typer.Argument(help="Time to maturity in years (T)")],
    rate: Annotated[float, typer.Argument(help="Risk-free rate (r)")],
    vol: Annotated[float, typer.Argument(help="Volatility (vol)")],
    option_type: Annotated[
        str, typer.Argument(help="Option type: call/c or put/p", parser=_parse_option_type)
    ],
    div_yield: Annotated[
        float, typer.Option("--div-yield", "-q", help="Continuous dividend yield")
    ] = 0.0,
) -> None:
    """Display Greeks with detailed explanations."""
    opt = cast(OptionType, option_type)
    params = OptionParams(
        spot=spot,
        strike=strike,
        time_to_maturity=time,
        risk_free_rate=rate,
        volatility=vol,
        option_type=opt,
        dividend_yield=div_yield,
    )

    result = price_option(params)
    g = result.greeks

    console.print(
        Panel.fit(
            f"[bold]Price:[/bold] {result.price:.6f}\n\n"
            f"[cyan]Delta:[/cyan] {g.delta:.6f}\n"
            f"  Rate of change of option price w.r.t. spot price.\n"
            f"  For {opt.value}: {'positive' if g.delta > 0 else 'negative'}.\n\n"
            f"[cyan]Gamma:[/cyan] {g.gamma:.6f}\n"
            f"  Rate of change of delta w.r.t. spot. Always positive.\n"
            f"  Measures convexity / delta hedging frequency.\n\n"
            f"[cyan]Vega:[/cyan] {g.vega:.6f}\n"
            f"  Sensitivity to 1% change in volatility. Always positive.\n"
            f"  Higher for ATM, longer-dated options.\n\n"
            f"[cyan]Theta:[/cyan] {g.theta:.6f}\n"
            f"  Time decay per day. Usually negative (option loses value).\n"
            f"  Accelerates near expiry for ATM options.\n\n"
            f"[cyan]Rho:[/cyan] {g.rho:.6f}\n"
            f"  Sensitivity to 1% change in risk-free rate.\n"
            f"  Positive for calls, negative for puts.",
            title=f"Greeks: {opt.value.capitalize()} K={strike} T={time} vol={vol:.1%}",
            border_style="blue",
        )
    )


@app.command(name="fit-surface")
def fit_surface(
    spot: Annotated[float, typer.Option("--spot", "-S", help="Current underlying price")],
    strikes: Annotated[
        str, typer.Option("--strikes", "-K", help="Comma-separated strikes, at least 4")
    ],
    prices: Annotated[str, typer.Option("--prices", "-p", help="Comma-separated market prices")],
    time: Annotated[float, typer.Option("--time", "-T", help="Time to maturity in years")],
    rate: Annotated[float, typer.Option("--rate", "-r", help="Risk-free rate")],
    option_type: Annotated[
        str, typer.Option("--type", "-t", help="call or put", parser=_parse_option_type)
    ] = "call",
    div_yield: Annotated[
        float, typer.Option("--div-yield", "-q", help="Continuous dividend yield")
    ] = 0.0,
    relaxed: Annotated[
        bool, typer.Option("--relaxed", help="Fit even if the butterfly check fails")
    ] = False,
    format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TEXT,
) -> None:
    """Fit an SVI volatility smile to a set of quoted option prices."""
    try:
        strike_list = [float(x) for x in strikes.split(",")]
        price_list = [float(x) for x in prices.split(",")]
    except ValueError:
        console.print("[red]Error:[/red] strikes and prices must be comma-separated numbers")
        raise typer.Exit(1) from None

    try:
        surface = VolSurface.from_market_prices(
            spot=spot,
            strikes=strike_list,
            prices=price_list,
            time_to_maturity=time,
            risk_free_rate=rate,
            dividend_yield=div_yield,
            option_type=cast(OptionType, option_type),
        )
        fit = surface.fit_relaxed() if relaxed else surface.fit()
    except SVIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if not relaxed:
            console.print("[dim]Hint: --relaxed will fit anyway, so you can inspect it.[/dim]")
        raise typer.Exit(1) from None

    curve = fit.curve

    if format == OutputFormat.JSON:
        console.print_json(
            json.dumps(
                {
                    "forward": surface.forward,
                    "parameters": {
                        "a": curve.a,
                        "b": curve.b,
                        "rho": curve.rho,
                        "m": curve.m,
                        "sigma": curve.sigma,
                    },
                    "rms_error": fit.rms_error,
                    "max_error": fit.max_error,
                    "arbitrage_free": {
                        "parameters": curve.parameters_are_valid(),
                        "butterfly": curve.butterfly_arbitrage_free(),
                    },
                    "strikes": list(fit.strikes),
                    "market_vols": list(fit.market_vols),
                    "model_vols": list(fit.model_vols),
                }
            )
        )
        return

    table = Table(title=f"SVI smile (forward {surface.forward:.4f}, T={time:g}y)")
    table.add_column("Strike", justify="right")
    table.add_column("Market Vol", justify="right")
    table.add_column("Model Vol", justify="right")
    table.add_column("Error", justify="right")

    for strike, market, model in zip(fit.strikes, fit.market_vols, fit.model_vols, strict=True):
        table.add_row(
            f"{strike:.2f}",
            f"{market:.4%}",
            f"{model:.4%}",
            f"{model - market:+.4%}",
        )
    console.print(table)

    console.print(
        Panel.fit(
            f"[bold]a     =[/bold] {curve.a:.6f}\n"
            f"[bold]b     =[/bold] {curve.b:.6f}\n"
            f"[bold]rho   =[/bold] {curve.rho:.6f}\n"
            f"[bold]m     =[/bold] {curve.m:.6f}\n"
            f"[bold]sigma =[/bold] {curve.sigma:.6f}\n"
            f"[bold]min w =[/bold] {curve.minimum_total_variance():.6f}\n\n"
            f"RMS error  : {fit.rms_error:.6%}\n"
            f"Max error  : {fit.max_error:.6%}\n"
            f"Butterfly  : {'free' if curve.butterfly_arbitrage_free() else 'VIOLATED'}",
            title="Fitted parameters",
            border_style="green" if curve.butterfly_arbitrage_free() else "red",
        )
    )


if __name__ == "__main__":
    app()
