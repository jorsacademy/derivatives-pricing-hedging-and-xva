"""Vasicek short-rate model and stylized swap valuation at reset dates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VasicekParams:
    mean_reversion: float = 0.35
    long_run_rate: float = 0.035
    volatility: float = 0.012
    initial_rate: float = 0.03

    def validate(self) -> None:
        if self.mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be nonnegative")


def vasicek_zcb_price(
    short_rate: float | np.ndarray,
    time: float,
    maturity: float,
    params: VasicekParams,
) -> np.ndarray:
    """Closed-form zero-coupon price P(t,T) under Vasicek."""
    params.validate()
    if time < 0 or maturity < time:
        raise ValueError("require 0 <= time <= maturity")
    if np.isclose(maturity, time):
        return np.ones_like(np.asarray(short_rate, dtype=float))

    a = params.mean_reversion
    b = params.long_run_rate
    sigma = params.volatility
    tau = maturity - time
    B = (1.0 - np.exp(-a * tau)) / a
    A = np.exp(
        (b - sigma**2 / (2.0 * a**2)) * (B - tau)
        - sigma**2 * B**2 / (4.0 * a)
    )
    return A * np.exp(-B * np.asarray(short_rate, dtype=float))


def simulate_vasicek_paths(
    params: VasicekParams,
    times: np.ndarray,
    n_paths: int = 20_000,
    seed: int = 7,
) -> np.ndarray:
    """Exact-transition simulation of Vasicek short-rate paths."""
    params.validate()
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional grid with at least two points")
    if not np.isclose(times[0], 0.0) or not np.all(np.diff(times) > 0):
        raise ValueError("times must start at zero and be strictly increasing")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    rates = np.empty((n_paths, len(times)), dtype=float)
    rates[:, 0] = params.initial_rate
    a = params.mean_reversion
    b = params.long_run_rate
    sigma = params.volatility

    for i, dt in enumerate(np.diff(times), start=1):
        decay = np.exp(-a * dt)
        mean = b + (rates[:, i - 1] - b) * decay
        variance = sigma**2 * (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
        rates[:, i] = mean + np.sqrt(max(variance, 0.0)) * rng.standard_normal(n_paths)

    return rates


def par_swap_rate_vasicek(
    params: VasicekParams,
    payment_times: np.ndarray,
) -> float:
    payment_times = np.asarray(payment_times, dtype=float)
    if len(payment_times) == 0 or (payment_times <= 0).any():
        raise ValueError("payment_times must be positive")
    discounts = np.array(
        [
            float(vasicek_zcb_price(params.initial_rate, 0.0, t, params))
            for t in payment_times
        ]
    )
    return float((1.0 - discounts[-1]) / discounts.sum())


def payer_swap_value_at_reset(
    short_rate: float | np.ndarray,
    time: float,
    payment_times: np.ndarray,
    fixed_rate: float,
    notional: float,
    params: VasicekParams,
) -> np.ndarray:
    """Stylized payer-swap value at reset dates.

    The floating leg is represented by 1-P(t,T_N), appropriate for the
    educational reset-date approximation used by the exposure engine.
    """
    if notional <= 0:
        raise ValueError("notional must be positive")
    payment_times = np.asarray(payment_times, dtype=float)
    remaining = payment_times[payment_times > time + 1e-12]
    r = np.asarray(short_rate, dtype=float)
    if len(remaining) == 0:
        return np.zeros_like(r)

    discount_matrix = np.stack(
        [vasicek_zcb_price(r, time, maturity, params) for maturity in remaining],
        axis=-1,
    )
    fixed_leg = fixed_rate * discount_matrix.sum(axis=-1)
    float_leg = 1.0 - discount_matrix[..., -1]
    return notional * (float_leg - fixed_leg)
