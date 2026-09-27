# Volatility Surface Calibration

A calibration project for two widely used smile/skew parameterizations.

The project deliberately separates quoted implied-volatility geometry from the
stochastic-volatility model used later in the Heston project.

## Raw SVI

For log-moneyness (k), total variance is represented as:

```text
w(k) = a + b [ rho (k-m) + sqrt((k-m)^2 + sigma^2) ]
```

The implementation calibrates `a, b, rho, m, sigma` by bounded nonlinear
least squares.

The project reports calibration RMSE and reproduces the fitted total-variance
slice on a dense log-moneyness grid.

## SABR

The second layer implements the Hagan lognormal SABR approximation.

For a fixed beta, the calibration solves for:

- alpha;
- rho;
- nu.

The runner constructs a synthetic strike smile, calibrates both models and
exports market-versus-model diagnostics.

## Run

```bash
python projects/volatility-surface-calibration/run.py
```

Generated outputs:

```text
outputs/
├── svi_fit.csv
├── sabr_fit.csv
└── calibration_summary.csv
```

## Scope

This is a transparent calibration layer, not a full arbitrage-free volatility
surface engine. It does not enforce calendar-spread constraints across
maturities or all static-arbitrage conditions for SVI. Those checks would be
required in a production volatility stack.
