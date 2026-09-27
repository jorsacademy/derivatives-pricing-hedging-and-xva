"""Black-Scholes pricing, Greeks, and implied-volatility utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float


def _validate(spot: float, strike: float, maturity: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    _validate(spot, strike, maturity, volatility)
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return float(d1), float(d2)


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    d1, d2 = _d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-dividend_yield * maturity)

    if option_type == "call":
        value = spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
    elif option_type == "put":
        value = strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return float(value)


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> OptionGreeks:
    d1, d2 = _d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    root_t = np.sqrt(maturity)
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-dividend_yield * maturity)

    gamma = df_q * norm.pdf(d1) / (spot * volatility * root_t)
    vega = spot * df_q * norm.pdf(d1) * root_t

    if option_type == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            -spot * df_q * norm.pdf(d1) * volatility / (2.0 * root_t)
            - rate * strike * df_r * norm.cdf(d2)
            + dividend_yield * spot * df_q * norm.cdf(d1)
        )
    elif option_type == "put":
        delta = df_q * (norm.cdf(d1) - 1.0)
        theta = (
            -spot * df_q * norm.pdf(d1) * volatility / (2.0 * root_t)
            + rate * strike * df_r * norm.cdf(-d2)
            - dividend_yield * spot * df_q * norm.cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return OptionGreeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    def objective(volatility: float) -> float:
        return black_scholes_price(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            option_type,
            dividend_yield,
        ) - market_price

    low, high = 1e-6, 5.0
    if objective(low) * objective(high) > 0:
        raise ValueError("market price is outside the supported implied-vol range")
    return float(brentq(objective, low, high, xtol=1e-12))
