# Heston Stochastic-Volatility Calibration

A full option-surface calibration example built around the Heston
characteristic function.

This project is intentionally more substantial than a closed-form pricing
notebook. It connects:

```text
stochastic-volatility dynamics
        ↓
characteristic function
        ↓
Carr-Madan damped Fourier pricing
        ↓
multi-strike / multi-maturity option surface
        ↓
bounded nonlinear calibration
        ↓
repricing diagnostics
```

## Model

Under the risk-neutral measure:

```text
dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^S

dv_t = kappa(theta-v_t) dt + sigma sqrt(v_t) dW_t^v

corr(dW_t^S, dW_t^v) = rho
```

The calibrated parameter vector is:

```text
(kappa, theta, sigma, rho, v0)
```

## Pricing

European calls are priced from the Heston characteristic function using a
Carr-Madan exponential damping transform and numerical Fourier integration.

The calibration layer minimizes price residuals across a grid of strikes and
maturities.

## Validation

The tests check two economically important properties:

- as vol-of-vol approaches zero with constant variance, Heston prices collapse
  toward Black-Scholes;
- calibration reprices a synthetic Heston surface to a small RMSE.

## Run

```bash
python projects/heston-calibration/run.py
```

Outputs:

```text
outputs/
├── synthetic_option_surface.csv
├── heston_fitted_surface.csv
└── heston_parameters.csv
```

## Limitations

The project uses synthetic option data and direct least-squares price errors.
Production calibration typically requires quote filtering, bid/ask weighting,
surface arbitrage controls, maturity-dependent parameter policy, numerical
stability diagnostics and market-data governance.
