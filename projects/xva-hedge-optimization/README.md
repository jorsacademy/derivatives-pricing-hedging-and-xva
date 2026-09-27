# XVA Hedge Optimization

A decision layer on top of the repository's XVA engine.

The project first measures finite-difference total-XVA sensitivities using
common Monte Carlo random numbers. It then chooses a synthetic hedge overlay
that reduces the largest normalized residual sensitivity under gross-notional
and instrument limits.

## Sensitivity layer

The default XVA risk vector contains finite-difference changes in total XVA
for small bumps to:

- counterparty hazard;
- funding spread;
- initial short rate;
- short-rate volatility.

Using the same Monte Carlo seed before and after each bump reduces simulation
noise in the finite difference.

## Hedge universe

The synthetic hedge set contains:

- counterparty CDS;
- funding-basis swap;
- rates swap overlay;
- swaption overlay.

Each hedge has a factor-sensitivity vector, unit cost and maximum notional.

## Optimization

Let b be the measured XVA sensitivity vector and H the hedge-sensitivity
matrix.

Residual risk is:

```text
r = b + Hx
```

Each factor is normalized by the magnitude of its unhedged sensitivity. The LP
minimizes:

```text
maximum normalized residual sensitivity
+
cost_weight * hedge cost
```

subject to signed instrument limits and a gross hedge budget.

This is a compact proxy for desk-level XVA hedge selection: not every
valuation adjustment is perfectly hedgeable, and the optimizer must allocate
a scarce hedge budget across market, credit and funding drivers.

## Run

```bash
python projects/xva-hedge-optimization/run.py
```

Outputs:

```text
outputs/
├── measured_xva_sensitivities.csv
├── xva_hedge_notionals.csv
├── residual_xva_sensitivities.csv
└── xva_hedge_summary.csv
```

## Limitations

The hedge-effect matrix is synthetic and linearized. Production XVA hedging
requires trade-level sensitivities, instrument market data, basis risk,
liquidity, regulatory capital, collateral effects, hedge accounting and
desk-specific funding policy.
