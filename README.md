# Derivatives Pricing, Hedging and XVA

Quantitative derivatives research focused on the decision layer around valuation, risk, counterparty exposure and hedge design.

The repository is deliberately not a collection of isolated pricing formulas. The common architecture is:

```text
market / curve state
        ↓
valuation and Greeks
        ↓
scenario or Monte Carlo exposure
        ↓
XVA / tail-risk measurement
        ↓
hedge optimization
        ↓
benchmark + reconciliation tests
```

## Project map

| Project | Core question | Methods | Status |
|---|---|---|---|
| [Curve Construction](projects/curve-construction/) | Build an internally consistent discount curve and reprice par instruments | Bootstrapping, discount factors, forward rates | Implemented |
| [Options, Volatility & Greeks](projects/options-volatility-and-greeks/) | Price vanilla options and recover implied volatility / risk sensitivities | Black-Scholes, Greeks, root finding | Implemented |
| [XVA Exposure Engine](projects/xva-exposure-engine/) | How much counterparty/funding valuation adjustment arises from simulated swap exposure? | Vasicek, Monte Carlo, EPE/ENE/PFE, CVA/DVA/FVA/MVA, wrong-way proxy | Flagship |
| [Robust Hedge Optimization](projects/robust-hedge-optimization/) | Which hedge overlay minimizes worst stress loss under cost and notional constraints? | LP, minimax optimization, scenario stress P&L | Flagship |

## Design principles

Every project is designed to contain:

1. an explicit quantitative finance problem;
2. a transparent mathematical model;
3. deterministic or seeded synthetic inputs;
4. executable Python code;
5. benchmark or reconciliation checks;
6. automated tests;
7. explicit limitations.

The emphasis is on models that are inspectable and falsifiable. Synthetic performance is not presented as evidence of trading profitability or production fitness.

## Installation

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run examples

```bash
python projects/curve-construction/run.py
python projects/options-volatility-and-greeks/run.py
python -m derivatives_xva.xva
python projects/xva-exposure-engine/run.py
python -m derivatives_xva.hedge
python projects/robust-hedge-optimization/run.py
```

## Repository structure

```text
projects/
├── curve-construction/
├── options-volatility-and-greeks/
├── xva-exposure-engine/
└── robust-hedge-optimization/

src/
└── derivatives_xva/
    ├── curves.py
    ├── options.py
    ├── rates.py
    ├── xva.py
    └── hedge.py

tests/
```

## Scope

This is a research and educational codebase. It is not a production pricing library, front-office risk engine, regulatory capital calculator, counterparty-credit platform, margin engine, hedge-accounting system, or trading system.

Production implementation would require market-data governance, multi-curve construction, collateral agreements, netting sets, calibration infrastructure, legal terms, model validation, numerical controls, trade lifecycle handling and institution-specific valuation policy.

## Disclaimer

Educational and research use only. Nothing in this repository is investment, trading, accounting, legal, credit or regulatory advice.
