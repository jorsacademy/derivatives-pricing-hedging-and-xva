# Mathematical Model

The Heston model is:

```text
dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^S

dv_t = kappa(theta-v_t)dt + sigma sqrt(v_t)dW_t^v

d<W^S,W^v>_t = rho dt
```

The implementation evaluates the characteristic function:

```text
phi(u;T) = E[exp(i u log S_T)]
```

in affine form:

```text
phi(u;T) = exp(iu log S_0 + C(u,T) + D(u,T) v_0)
```

European calls are recovered using the Carr-Madan damped transform. For
damping parameter alpha > 0:

```text
C(K)
=
exp(-alpha log K) / pi
*
integral_0^infinity
Re[
    exp(-iv log K)
    exp(-rT) phi(v-(alpha+1)i)
    /
    (alpha^2 + alpha - v^2 + i(2alpha+1)v)
] dv
```

Calibration solves:

```text
min_theta
    sum_j (C_model(K_j,T_j; theta) - C_market_j)^2
```

subject to bounded economically admissible parameter ranges.

The project does not claim uniqueness of Heston parameters. Surface
calibration can be ill-conditioned, and production calibration requires
additional regularization and stability controls.
