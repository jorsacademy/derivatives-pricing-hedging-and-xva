"""Seeded Monte Carlo pricing for path-dependent equity derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloPrice:
    price: float
    standard_error: float


def simulate_gbm_paths(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int = 7,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate GBM paths with optional antithetic variates."""
    if spot <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, volatility and maturity must be positive")
    if n_steps < 1 or n_paths < 2:
        raise ValueError("n_steps>=1 and n_paths>=2 are required")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps

    if antithetic:
        half = (n_paths + 1) // 2
        base = rng.standard_normal((half, n_steps))
        shocks = np.vstack([base, -base])[:n_paths]
    else:
        shocks = rng.standard_normal((n_paths, n_steps))

    increments = (
        (rate - dividend_yield - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * shocks
    )
    log_relative = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_relative)
    return paths


def _discounted_result(payoff: np.ndarray, rate: float, maturity: float) -> MonteCarloPrice:
    discounted = np.exp(-rate * maturity) * np.asarray(payoff, dtype=float)
    return MonteCarloPrice(
        price=float(discounted.mean()),
        standard_error=float(discounted.std(ddof=1) / np.sqrt(len(discounted))),
    )


def european_call_mc(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    n_steps: int = 64,
    n_paths: int = 20_000,
    seed: int = 7,
) -> MonteCarloPrice:
    paths = simulate_gbm_paths(
        spot,
        rate,
        dividend_yield,
        volatility,
        maturity,
        n_steps,
        n_paths,
        seed,
    )
    return _discounted_result(
        np.maximum(paths[:, -1] - strike, 0.0),
        rate,
        maturity,
    )


def arithmetic_asian_call_mc(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    n_steps: int = 64,
    n_paths: int = 20_000,
    seed: int = 7,
) -> MonteCarloPrice:
    """Arithmetic-average Asian call using monitored post-inception prices."""
    paths = simulate_gbm_paths(
        spot,
        rate,
        dividend_yield,
        volatility,
        maturity,
        n_steps,
        n_paths,
        seed,
    )
    average = paths[:, 1:].mean(axis=1)
    return _discounted_result(
        np.maximum(average - strike, 0.0),
        rate,
        maturity,
    )


def up_and_out_call_mc(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    n_steps: int = 128,
    n_paths: int = 30_000,
    seed: int = 7,
) -> MonteCarloPrice:
    """Discretely monitored up-and-out call."""
    if barrier <= spot:
        return MonteCarloPrice(price=0.0, standard_error=0.0)

    paths = simulate_gbm_paths(
        spot,
        rate,
        dividend_yield,
        volatility,
        maturity,
        n_steps,
        n_paths,
        seed,
    )
    knocked_out = paths[:, 1:].max(axis=1) >= barrier
    payoff = np.where(
        knocked_out,
        0.0,
        np.maximum(paths[:, -1] - strike, 0.0),
    )
    return _discounted_result(payoff, rate, maturity)
