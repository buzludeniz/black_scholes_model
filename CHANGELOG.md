# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dependabot configuration for pip and GitHub Actions, weekly, with minor and
  patch updates grouped.
- `CHANGELOG.md` and `RELEASING.md` (one-time PyPI trusted-publisher setup and
  the tagging steps).
- Project URLs (Changelog, Issues) in the package metadata.
- `black_scholes.term_structure.TermStructure`: a multi-expiry surface built
  from SVI slices by expiry (`from_slices`, or `from_surfaces` to fit each
  `VolSurface`). Every adjacent pair of expiries is checked for calendar
  arbitrage, reusing `SVICurve.calendar_arbitrage_free` for crossings and adding
  an ordering check for a later slice lying wholly below an earlier one;
  `calendar_violations()` lists the offending pairs. Total implied variance is
  interpolated linearly in time between expiries, and `implied_volatility(k, T)`
  is available for any `T` inside the quoted range. Maturities outside it are
  rejected rather than extrapolated. `butterfly_arbitrage_free(T)` checks an
  interpolated smile.
- American option pricing in `black_scholes.american`: a Cox-Ross-Rubinstein
  binomial tree with early exercise and continuous dividend yield
  (`binomial_price`, also able to price the European option on the same tree),
  the Barone-Adesi and Whaley (1987) quadratic approximation
  (`barone_adesi_whaley_price`), and `american_price(method="crr" | "baw")`.
  Tested against Hull's published tree values and Haug's BAW values.
- `bs american` command: CRR and/or BAW price next to the European price, with
  the early-exercise premium, `--steps`, `--method` and `--format json`.

### Changed
- README: corrected the install route, the SVI formula, and the vega-weighting
  note.

## [0.2.0] - not yet released

`0.2.0` is the version declared in `pyproject.toml` since the first commit, and
no `v0.2.0` tag exists yet, so everything in the repository's history is part of
this release. Set the release date here when the tag is cut.

### Added
- Black-Scholes European option pricing with the full Greek set, implied
  volatility inversion, and a vectorised Monte Carlo reference implementation.
- `bs` command line interface (Typer) and `bs-gui` Tkinter front end.
- SVI volatility surface fitting (`fit_svi`, `VolSurface`), with the
  arbitrage-free parameter, butterfly and calendar-spread checks from Gatheral
  and Jacquier (arXiv:1204.0646), and SVI-Jump-Wings parameters.
- `bs fit-surface` command, with text or JSON output and `--relaxed` to inspect
  a fit the guards rejected.
- Vega-weighted least squares for SVI fitting: `weights` option (`"vega"` by
  default, `"uniform"` for the previous behaviour) on `fit_svi()`,
  `VolSurface.fit()` and `VolSurface.fit_relaxed()`, and `--weights/-w` on the
  CLI. Reported `rms_error` and `max_error` stay in raw implied-vol terms.
- GitHub Actions CI (Python 3.10 to 3.13 on Linux and Windows, lint, type check,
  wheel and sdist install smoke tests, 85% coverage gate) and a PyPI trusted
  publishing workflow for `v*` tags.
- `demo_gui.py`, which drives the real GUI widgets and prints the display.
- Packaging tests pinning the version, distribution name, README install line
  and entry points.

### Changed
- The distribution is named `black_scholes_model` (the import name stays
  `black_scholes`), so the documented `pip install` line can resolve.
- The coverage gate moved from pytest `addopts` into CI so partial test runs do
  not fail.
- The `--relaxed` hint on `fit-surface` appears only for failures it can bypass.

### Fixed
- Butterfly arbitrage condition used `w'(k)` where Gatheral and Jacquier's
  equation (2.1) has `w'(k)^2`; about 6% of sampled curves were wrongly called
  safe.
- The SVI fitter could return an arbitrage-free curve that missed the quotes by
  hundreds of volatility points and report success; an error tolerance
  (default 0.10) now bounds the worst deviation.
- NaN and infinite inputs are rejected everywhere (previously pricing returned
  `nan`); surface construction validates empty, mismatched, duplicate and
  non-positive quotes.
- CLI errors from malformed input or domain failures print one `Error:` line and
  exit 1, without a traceback.
- The calendar-arbitrage check no longer trusts the caller's grid: it locates
  extrema of the slice difference analytically and scans on its own grid of at
  least 513 points, so a crossing inside a single grid cell is detected.
- `implied_volatility` returned an arbitrary interior volatility for prices
  carrying no time value (for example a deep out-of-the-money short-dated put);
  the residual test now applies only when the bracket separates prices, and the
  noise floor scales with `max(S, K)`.
- `implied_volatility` rejects a non-finite `max_iter` with `ValueError`.

[Unreleased]: https://github.com/buzludeniz/black_scholes_model/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/buzludeniz/black_scholes_model/releases/tag/v0.2.0
