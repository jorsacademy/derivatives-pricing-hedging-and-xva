# Mathematical Model

## Short-rate process

The risk-factor process is Vasicek:

```text
dr_t = a(b-r_t)dt + sigma dW_t
```

with exact discrete transition over `Delta t`:

```text
E[r_{t+Delta}|r_t] = b + (r_t-b)e^{-a Delta}
Var[r_{t+Delta}|r_t] = sigma^2 (1-e^{-2a Delta}) / (2a)
```

## Swap exposure

At synthetic reset dates the payer-swap value is approximated by:

```text
V_t = N * [(1-P(t,T_N)) - K * sum_i P(t,T_i)]
```

for remaining payment dates `T_i > t`.

Scenario exposure is:

```text
E_t^+ = max(V_t, 0)
E_t^- = max(-V_t, 0)
```

The profile reports sample means of these quantities plus a high quantile of positive exposure.

## CVA and DVA

With deterministic hazard rate `lambda_c` and recovery `R_c`:

```text
CVA = (1-R_c) * sum_i DF_i * EPE_i * DeltaPD_i
```

DVA is calculated analogously from expected negative exposure and own default probability.

## FVA and MVA

The educational funding adjustments use:

```text
FVA = sum_i DF_i * EPE_i * funding_spread * Delta t_i
MVA = sum_i DF_i * IM_i  * im_funding_spread * Delta t_i
```

where `IM_i` is a PFE-based proxy rather than a regulatory or CCP initial-margin model.

## Wrong-way proxy

Positive exposures may be reweighted by:

```text
w_path,t ∝ exp(beta * z(r_path,t))
```

where `z(.)` is the cross-sectional standardized short rate. This preserves the Monte Carlo architecture while making the dependency assumption inspectable.
