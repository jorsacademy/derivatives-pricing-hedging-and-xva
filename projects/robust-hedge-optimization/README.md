# Robust Hedge Optimization

A minimax hedge-design problem for a synthetic derivatives portfolio.

The project starts from full-revaluation stress P&L vectors for an unhedged portfolio and several candidate hedge instruments. The optimizer chooses signed hedge notionals to reduce the worst scenario loss while paying an explicit linear hedge-cost penalty.

## Stress set

The default synthetic stress set contains:

- equity crash + volatility spike;
- moderate equity selloff;
- pure volatility spike;
- moderate rally;
- melt-up + volatility crush.

Candidate overlays are represented by scenario P&L vectors for a put spread, vega hedge and futures hedge.

## Optimization

The LP minimizes:

```text
worst residual scenario loss
+
cost_weight * hedge carry / implementation cost
```

subject to:

- instrument-level signed-notional limits;
- total gross-notional budget;
- one common hedge portfolio across all scenarios.

This is deliberately different from matching a single delta or vega target. The optimizer acts on the joint stress surface.

## Run

```bash
python -m derivatives_xva.hedge
python projects/robust-hedge-optimization/run.py
```

Generated outputs:

```text
outputs/
├── hedge_notionals.csv
├── residual_stress_pnl.csv
└── hedge_summary.csv
```

## Next research extensions

Natural extensions include CVaR instead of pure minimax loss, nonlinear option repricing inside scenario generation, liquidity-dependent transaction costs, multi-period hedge rebalancing, capital constraints and XVA sensitivities as hedge targets.
