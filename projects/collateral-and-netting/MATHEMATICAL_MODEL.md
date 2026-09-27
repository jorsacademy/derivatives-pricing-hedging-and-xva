# Mathematical Model

Let V_{p,t,j} be the simulated mark-to-market of trade j on path p at time t.

Legal netting gives:

```text
N_{p,t} = sum_j V_{p,t,j}
```

The desired signed variation-margin collateral is:

```text
C*_{p,t}
=
N_{p,t} - threshold_receive          if N_{p,t} > threshold_receive

N_{p,t} + threshold_post             if N_{p,t} < -threshold_post

0                                    otherwise
```

A transfer is executed only if:

```text
|C*_{p,t} - C_{p,t-1}| >= MTA
```

and the collateral call can use a lagged reference MtM to represent a discrete
margin-period-of-risk proxy.

Unsecured mark-to-market is:

```text
U_{p,t} = N_{p,t} - C_{p,t}
```

Exposure measures are then:

```text
EPE_t = E[max(U_t,0)]
ENE_t = E[max(-U_t,0)]
PFE_t = quantile_q(max(U_t,0))
```

The implementation is pathwise and therefore keeps threshold, MTA and lag
effects inside each scenario rather than applying a portfolio-level haircut
after exposure aggregation.
