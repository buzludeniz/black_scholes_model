# black_scholes_model

European option pricing built on the Black-Scholes-Merton formula, with the full
Greek set, implied-volatility inversion, and a vectorised Monte Carlo reference
implementation for cross-checking.

- **Correct by construction.** Put-call parity is exact to 1e-10, Greeks are
  validated against finite differences, and every public function carries
  runnable doctests.
- **Three front ends.** Python API, a `bs` CLI, and a Tkinter GUI.
- **Honest numbers.** Every function returns a native `float`, not a NumPy
  scalar, so results serialise cleanly.

## Install

```bash
pip install black_scholes_model
```

From a checkout, with the development extras:

```bash
git clone https://github.com/buzludeniz/black_scholes_model.git
cd black_scholes_model
pip install -e ".[dev]"
```

Requires Python 3.10 or newer.

## Quick start

```python
from black_scholes import OptionParams, OptionType, price_option, implied_volatility

params = OptionParams(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
    option_type=OptionType.CALL,
)

result = price_option(params)
print(f"Price: {result.price:.4f}")  # 10.4506
print(f"Delta: {result.greeks.delta:.4f}")  # 0.6368

# Recover the volatility the market is pricing in.
iv = implied_volatility(result.price, params)
print(f"Implied vol: {iv:.2%}")  # 20.00%
```

## Command line

```bash
# Price with Greeks
bs price 100 100 1 0.05 0.2 call

# Machine-readable output
bs price 100 100 1 0.05 0.2 call --format json

# Cross-check against a simulation
bs price 100 100 1 0.05 0.2 call --mc

# Implied volatility from a market price
bs iv 10.450584 100 100 1 0.05 call

# Strike-by-strike surface
bs surface --spot 100 --time 1 --rate 0.05 --vol 0.2 --strikes "80,90,100,110,120"

# Greeks with explanations
bs greeks 100 100 1 0.05 0.2 call
```

`call` also accepts `c` and `put` accepts `p`. Use `-q/--div-yield` for a
continuous dividend yield.

## Graphical interface

```bash
bs-gui
```

Enter the spot, strike, maturity, rate, volatility, and dividend yield; pick call
or put; press **Calculate Price & Greeks**. Tick the Monte Carlo box to see the
simulation cross-check the closed form, or **Implied Volatility** to solve for the
volatility behind a market price.

## API

| Function | Purpose |
|----------|---------|
| `black_scholes_price(params)` | Closed-form price |
| `black_scholes_greeks(params)` | Delta, Gamma, Vega, Theta, Rho |
| `price_option(params)` | Price and Greeks together |
| `implied_volatility(price, params)` | Invert price to volatility |
| `arbitrage_bounds(params)` | No-arbitrage price interval |
| `monte_carlo_price(params, ...)` | Simulated price with standard error |

Greek units: Vega is per 1% volatility move, Theta is per calendar day, Rho is per
1% rate move. Gamma and Vega are identical for calls and puts.

### Model

```
Call:  S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
Put:   K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
d2 = d1 - σ·√T
```

This is the European option model under geometric Brownian motion with a
constant volatility and a continuous dividend yield.

## Development

```bash
pip install -e ".[dev]"

pytest                       # tests plus doctests, with an 85% coverage gate
ruff check . && ruff format --check .
mypy src/black_scholes
python -m build              # wheel and sdist

python check_layout.py       # GUI layout audit, exits non-zero if clipped
python launch_gui.py         # run the GUI
python examples/basic_usage.py
```

Current state: 187 tests pass, 88% branch coverage (core module at 99%), with
ruff and mypy clean.

## Known gaps

Honest inventory of what is not done:

- **GUI coverage is 73%.** Widget construction and the market-price dialog are not
  unit tested. `check_layout.py` covers geometry instead.
- **No CI.** Tests run locally only; there is no GitHub Actions workflow.
- **Not on PyPI.** `python -m build` produces valid artifacts, but the package is
  not published, so `pip install black_scholes_model` will not yet resolve.
- **No volatility surface fitting.** `implied_volatility` reads one point at a
  time rather than fitting a parameterisation such as SVI.
- **Single-asset only.** No basket, quanto, or correlation handling.

## Limitations

- European exercise only. American and Bermudan options are not implemented.
- Constant volatility. There is no term structure or smile fitting, though
  `implied_volatility` can be run per strike to read one off a surface.
- `implied_volatility` needs a no-arbitrage price and a volatility bracket that
  contains the answer. Prices at the theoretical bound require a wider bracket,
  and anything outside the bounds is rejected rather than extrapolated.

## License

MIT. See [LICENSE](LICENSE).