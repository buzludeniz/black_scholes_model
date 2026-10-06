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

The distribution is `black_scholes_model` but the import name is shorter,
`black_scholes`:

```python
import black_scholes
```

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

## Volatility surfaces

Real markets do not quote a flat volatility, they quote a smile. The
`black_scholes.surface` module fits that smile from market prices using the SVI
parameterisation, and refuses to return a fit that would let someone price
arbitrage out of it.

```python
from black_scholes.surface import VolSurface

surface = VolSurface.from_market_prices(
    spot=100.0,
    strikes=[70.0, 90.0, 100.0, 110.0, 150.0],
    prices=[34.3824, 18.0797, 11.7507, 7.4028, 1.3415],
    time_to_maturity=1.0,
    risk_free_rate=0.05,
)

fit = surface.fit()
print(fit.curve.a, fit.curve.b, fit.curve.rho)  # SVI parameters
print(fit.rms_error, fit.max_error)  # fit quality
```

From the shell:

```bash
bs fit-surface --spot 100 --time 1 --rate 0.05 \
  --strikes "70,90,100,110,150" \
  --prices  "34.3824,18.0797,11.7507,7.4028,1.3415"
```

### Why SVI

Total implied variance is written as a function of log-moneyness `k = ln(K/F)`:

```
w(k) = a + b·(ρ·(k − m) + √((k − m)² + σ²))
```

Five parameters, and the shape stays economically sensible where a polynomial
would eventually turn over: `b` sets the smile, `ρ` the skew, `σ` the wings.

### Three guards, because a fit that looks valid can still be useless

1. **Parameter arbitrage**, closed form: `b ≥ 0`, `|ρ| < 1`, `σ > 0`, and a
   non-negative minimum total variance.
2. **Butterfly arbitrage**, Gatheral's density condition, checked numerically:
   `(1 − k·w′/2w)² − w′/4·(1/w + 1/4) + w″/2 ≥ 0`.
3. **Error tolerance**: SVI cannot fit every quoted shape. Without this bound the
   optimiser can return a curve that passes both arbitrage checks yet misses
   the data by tens of volatility points, and report success.

`fit_relaxed()` lifts guards 2 and 3 so a rejected fit can be inspected instead
of vanishing.

## Development

```bash
pip install -e ".[dev]"

pytest                       # tests plus doctests, with an 85% coverage gate
ruff check . && ruff format --check .
mypy src/black_scholes
python -m build              # wheel and sdist

python check_layout.py       # GUI layout audit, exits non-zero if clipped
python demo_gui.py           # drive the real GUI widgets and print the display
python launch_gui.py         # run the GUI
python examples/basic_usage.py
```

Current state: 349 tests pass, 97% branch coverage, with ruff and mypy clean.
Every source module is above 91%; the GUI is at 100%.

## Known gaps

Honest inventory of what is still not done:

- **Not on PyPI.** `python -m build` produces valid artifacts and a
  Trusted-Publishing workflow is committed, but the package is not published, so
  `pip install black_scholes_model` will not resolve until the PyPI project is
  created and linked to the workflow.
- **Calendar arbitrage is only checked pairwise.** `SVICurve` offers
  `calendar_arbitrage_free` for two slices, but there is no term-structure
  object that validates a whole surface at once.
- **Butterfly arbitrage is a grid check, not a proof.** No closed form exists for
  raw SVI, and SSVI's closed-form conditions come from a stricter
  parameterisation that cannot fit every quoted smile. A finer grid can find
  more violations, never fewer.
- **No SABR or eSSVI.** SVI only.
- **Single-expiry fitting only.** Multi-expiry surfaces are assembled by the
  caller, one slice per expiry.
- **Unweighted least squares.** The fit weights every strike equally rather
  weighting by vega, which is what most practitioners do.
- **No American options.** European exercise only.

## Limitations

- European exercise only. American and Bermudan options are not implemented.
- `black_scholes_price` assumes a single flat volatility. For a real smile, price
  each strike with its own implied volatility or use the fitted surface.
- `implied_volatility` needs a no-arbitrage price and a volatility bracket that
  contains the answer. Prices at the theoretical bound require a wider bracket,
  and anything outside the bounds is rejected rather than extrapolated.
- The SVI fit is a least-squares approximation. SVI cannot represent every
  quoted shape, and the error bound exists so that failure is visible rather
  than silent.

## License

MIT. See [LICENSE](LICENSE).