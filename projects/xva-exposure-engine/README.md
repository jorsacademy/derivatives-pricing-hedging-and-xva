# XVA Exposure Engine

A Monte Carlo counterparty-exposure engine for a synthetic payer interest-rate swap.

This is the first flagship project in the repository. The system links a short-rate model, derivative valuation, future exposure distributions and valuation adjustments in one reproducible pipeline.

## Architecture

```text
Vasicek short-rate model
        ↓
Monte Carlo rate paths
        ↓
payer-swap mark-to-market at reset dates
        ↓
EPE / ENE / PFE / IM proxy
        ↓
CVA / DVA / FVA / MVA
        ↓
wrong-way-risk sensitivity
```

## Rates model

The short rate follows a Vasicek process:

```text
dr_t = a(b-r_t)dt + sigma dW_t
```

The implementation uses the exact Gaussian transition and the closed-form Vasicek zero-coupon bond price.

The swap fixed rate is set to the model-implied par rate at inception. Future values are evaluated at synthetic reset dates using remaining fixed cash flows and the floating-leg reset-date approximation.

## Exposure metrics

For every future date the simulation reports:

- expected positive exposure (EPE);
- expected negative exposure (ENE);
- potential future exposure (PFE);
- a PFE-based initial-margin proxy.

## XVA layer

The engine computes transparent synthetic approximations for:

- CVA from counterparty hazard and EPE;
- DVA from own hazard and ENE;
- FVA from expected positive funding exposure;
- MVA from the initial-margin proxy.

A simple wrong-way-risk proxy reweights positive exposure toward high-rate states. For a payer swap, positive mark-to-market tends to coincide with higher rates, so a positive dependency parameter can increase CVA.

The model is intentionally explicit about this approximation; it is not a production wrong-way-risk model.

## Run

```bash
python -m derivatives_xva.xva
python projects/xva-exposure-engine/run.py
```

Generated outputs:

```text
outputs/
├── exposure_profile.csv
└── xva_summary.csv
```

## Limitations

The engine uses one synthetic swap, one-factor Vasicek dynamics, constant marginal hazard rates, no collateral agreement, no netting set, no stochastic credit spread process and no legal closeout model.

Production XVA requires trade-level netting, CSA terms, collateral simulation, calibrated market dynamics, credit curves, wrong-way-risk governance, funding policy, initial-margin methodology and independent model validation.
