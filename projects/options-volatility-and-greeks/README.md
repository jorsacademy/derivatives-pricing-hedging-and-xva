# Options, Volatility and Greeks

A transparent vanilla-options layer used for pricing and risk diagnostics.

The module provides:

- Black-Scholes call/put valuation;
- delta, gamma, vega and theta;
- implied-volatility inversion with a bounded root solver;
- put-call-parity and implied-volatility recovery tests.

The goal is not to present Black-Scholes as a flagship result. This is infrastructure for later volatility-surface, exotic-pricing and hedge-optimization layers.

Run:

```bash
python projects/options-volatility-and-greeks/run.py
```
