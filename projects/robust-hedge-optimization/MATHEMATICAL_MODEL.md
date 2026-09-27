# Mathematical Model

Let `p_s` be the unhedged portfolio P&L in stress scenario `s` and `h_sj` the P&L contribution of one unit of hedge `j` in scenario `s`.

Signed hedge notional is split into positive and negative parts:

```text
x_j = x_j^+ - x_j^-
```

Residual scenario P&L is:

```text
R_s = p_s + sum_j h_sj x_j
```

Introduce `z >= 0` as the maximum residual loss. The LP is:

```text
minimize
    z + lambda * sum_j c_j (x_j^+ + x_j^-)

subject to
    -R_s <= z                          for every scenario s
    sum_j (x_j^+ + x_j^-) <= G
    0 <= x_j^+, x_j^- <= U_j
```

where `c_j` is the hedge-cost coefficient, `G` the gross hedge budget and `U_j` instrument-level bounds.

Scenario P&Ls are synthetic in this repository. In a production stack they would come from a full repricing engine under governed market stresses.
