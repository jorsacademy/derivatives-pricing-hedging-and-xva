# Curve Construction

A compact annual-pay par-swap bootstrap used as the rates foundation for the repository.

The project converts a sequence of synthetic par swap rates into discount factors and continuously compounded zero rates. It then reprices every input swap as a validation step.

For annual coupons and maturity `n`, the terminal discount factor is solved recursively from:

```text
1 = c_n * sum_{i=1..n} P(0,i) + P(0,n)
```

The implementation is intentionally transparent. It is not a production multi-curve OIS/IBOR bootstrap and does not model day-count conventions, stubs, calendars, futures convexity or collateral-specific discounting.

Run:

```bash
python projects/curve-construction/run.py
```
