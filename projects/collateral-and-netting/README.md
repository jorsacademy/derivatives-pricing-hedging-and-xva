# Collateral and Netting-Set Exposure

A synthetic CSA / counterparty-risk project that moves the repository from
standalone trade exposure toward netting-set economics.

## Architecture

```text
multiple correlated trade MtM paths
        ↓
legal netting set
        ↓
bilateral CSA thresholds
        ↓
minimum transfer amount
        ↓
margin-call lag / margin period proxy
        ↓
collateral balance
        ↓
unsecured EPE / ENE / PFE
```

## CSA mechanics

The collateral balance is signed:

- positive: collateral held by the dealer;
- negative: collateral posted by the dealer.

If positive net MtM exceeds the receive threshold, the desired collateral is
the excess over that threshold.

If negative net MtM exceeds the posting threshold, the desired balance is
negative collateral posted.

Transfers smaller than the minimum transfer amount are ignored. A discrete
margin lag creates residual exposure even when the thresholds are low.

## Benchmark

The runner compares:

```text
uncollateralized netting set
vs.
CSA-collateralized netting set
```

and reports mean EPE, terminal PFE and collateral usage.

## Run

```bash
python projects/collateral-and-netting/run.py
```

Outputs:

```text
outputs/
├── uncollateralized_profile.csv
├── collateralized_profile.csv
└── collateral_summary.csv
```

## Limitations

This is not a legal-document or production margin engine. It omits cure
periods, collateral haircuts, eligible collateral schedules, dispute
mechanics, interest on collateral, initial-margin segregation, closeout and
multi-currency collateral optimization.
