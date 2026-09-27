# Exotic Monte Carlo

A reproducible Monte Carlo layer for path-dependent equity derivatives.

The project currently prices:

- a European call as a benchmark;
- an arithmetic-average Asian call;
- a discretely monitored up-and-out call.

The simulation uses GBM with antithetic variates and reports Monte Carlo
standard errors explicitly.

The European benchmark is reconciled against the analytical Black-Scholes
price. This provides a numerical validation path before applying the same
simulation machinery to path-dependent payoffs.

## Run

```bash
python projects/exotic-monte-carlo/run.py
```

Outputs:

```text
outputs/
└── exotic_price_comparison.csv
```

## Limitations

The current layer does not yet include local/stochastic volatility, Brownian
bridge barrier correction, quasi-Monte Carlo, Longstaff-Schwartz early
exercise, adjoint Greeks or GPU acceleration. Those are natural extensions,
not hidden assumptions.
