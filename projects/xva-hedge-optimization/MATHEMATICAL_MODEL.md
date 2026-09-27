# Mathematical Model

Let b_i be the unhedged total-XVA sensitivity to risk factor i and H_{ij} the
sensitivity contribution of one unit of hedge j.

Signed hedge notional is represented as:

```text
x_j = x_j^+ - x_j^-
```

Residual sensitivity is:

```text
r_i = b_i + sum_j H_{ij} x_j
```

To compare heterogeneous factors, define:

```text
s_i = max(|b_i|, 1)
```

and introduce z as the maximum normalized residual.

The LP is:

```text
minimize
    z + lambda sum_j c_j (x_j^+ + x_j^-)

subject to
    +r_i / s_i <= z                 for every factor i
    -r_i / s_i <= z                 for every factor i

    sum_j (x_j^+ + x_j^-) <= G

    0 <= x_j^+, x_j^- <= U_j
    z >= 0
```

The base sensitivity vector is measured from the Monte Carlo XVA engine with
common-random-number finite differences.

The hedge matrix remains synthetic in this research repository so that the
optimization architecture is inspectable without pretending to reproduce a
specific dealer book.
