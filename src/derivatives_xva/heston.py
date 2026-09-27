"""Heston characteristic-function pricing and synthetic calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import least_squares


@dataclass(frozen=True)
class HestonParams:
    kappa: float = 1.5
    theta: float = 0.04
    sigma: float = 0.55
    rho: float = -0.60
    v0: float = 0.045

    def validate(self) -> None:
        if self.kappa <= 0 or self.theta <= 0 or self.sigma <= 0 or self.v0 <= 0:
            raise ValueError("Heston kappa, theta, sigma and v0 must be positive")
        if not -1 < self.rho < 1:
            raise ValueError("Heston rho must lie in (-1,1)")


@dataclass(frozen=True)
class HestonCalibrationResult:
    params: HestonParams
    rmse: float
    fitted: pd.DataFrame


def heston_characteristic_function(
    u: np.ndarray | complex,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    params: HestonParams,
) -> np.ndarray:
    """Risk-neutral characteristic function of log S_T."""
    params.validate()
    if spot <= 0 or maturity <= 0:
        raise ValueError("spot and maturity must be positive")

    u = np.asarray(u, dtype=complex)
    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    iu = 1j * u

    d = np.sqrt(
        (kappa - rho * sigma * iu) ** 2
        + sigma**2 * (iu + u**2)
    )
    g = (
        kappa - rho * sigma * iu - d
    ) / (
        kappa - rho * sigma * iu + d
    )
    exp_dt = np.exp(-d * maturity)

    C = (
        (rate - dividend_yield) * iu * maturity
        + (kappa * theta / sigma**2)
        * (
            (kappa - rho * sigma * iu - d) * maturity
            - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )
    D = (
        (kappa - rho * sigma * iu - d) / sigma**2
        * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    )
    return np.exp(iu * np.log(spot) + C + D * v0)


def heston_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: HestonParams,
    dividend_yield: float = 0.0,
    *,
    damping: float = 1.5,
    integration_limit: float = 120.0,
    integration_points: int = 1800,
) -> float:
    """Carr-Madan damped Fourier price of a European call."""
    if spot <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("spot, strike and maturity must be positive")
    if damping <= 0 or integration_limit <= 0 or integration_points < 200:
        raise ValueError("invalid Fourier integration settings")

    v = np.linspace(1e-8, integration_limit, integration_points)
    shifted = v - 1j * (damping + 1.0)
    phi = heston_characteristic_function(
        shifted,
        spot,
        maturity,
        rate,
        dividend_yield,
        params,
    )
    denominator = (
        damping**2
        + damping
        - v**2
        + 1j * (2.0 * damping + 1.0) * v
    )
    psi = np.exp(-rate * maturity) * phi / denominator
    log_strike = np.log(strike)
    integrand = np.real(np.exp(-1j * v * log_strike) * psi)
    integral = trapezoid(integrand, v)
    price = np.exp(-damping * log_strike) * integral / np.pi
    return float(max(price, 0.0))


def calibrate_heston(
    market: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> HestonCalibrationResult:
    """Calibrate all five Heston parameters to synthetic option prices.

    market must contain columns: strike, maturity, call_price.
    """
    required = {"strike", "maturity", "call_price"}
    if not required.issubset(market.columns) or len(market) < 8:
        raise ValueError("market must contain at least eight strike/maturity/price rows")
    if (market["strike"] <= 0).any() or (market["maturity"] <= 0).any():
        raise ValueError("strikes and maturities must be positive")

    observations = market.loc[:, ["strike", "maturity", "call_price"]].to_numpy(dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        params = HestonParams(
            kappa=float(x[0]),
            theta=float(x[1]),
            sigma=float(x[2]),
            rho=float(x[3]),
            v0=float(x[4]),
        )
        return np.array(
            [
                heston_call_price(
                    spot,
                    strike,
                    maturity,
                    rate,
                    params,
                    dividend_yield,
                    integration_limit=100.0,
                    integration_points=1200,
                )
                - call_price
                for strike, maturity, call_price in observations
            ]
        )

    result = least_squares(
        residual,
        x0=np.array([1.20, 0.05, 0.50, -0.30, 0.05]),
        bounds=(
            np.array([0.05, 0.005, 0.05, -0.95, 0.005]),
            np.array([5.00, 0.25, 2.00, 0.95, 0.25]),
        ),
        max_nfev=160,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError(f"Heston calibration failed: {result.message}")

    params = HestonParams(
        kappa=float(result.x[0]),
        theta=float(result.x[1]),
        sigma=float(result.x[2]),
        rho=float(result.x[3]),
        v0=float(result.x[4]),
    )
    fitted = market.copy()
    fitted["model_price"] = [
        heston_call_price(
            spot,
            float(row.strike),
            float(row.maturity),
            rate,
            params,
            dividend_yield,
        )
        for row in fitted.itertuples(index=False)
    ]
    fitted["pricing_error"] = fitted["model_price"] - fitted["call_price"]
    rmse = float(np.sqrt(np.mean(fitted["pricing_error"].to_numpy() ** 2)))
    return HestonCalibrationResult(params=params, rmse=rmse, fitted=fitted)


def synthetic_heston_market(
    spot: float = 100.0,
    rate: float = 0.03,
    dividend_yield: float = 0.0,
    params: HestonParams | None = None,
) -> pd.DataFrame:
    """Return a deterministic synthetic calibration surface."""
    p = params or HestonParams()
    rows = []
    for maturity in (0.5, 1.0, 2.0):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            rows.append(
                {
                    "strike": strike,
                    "maturity": maturity,
                    "call_price": heston_call_price(
                        spot,
                        strike,
                        maturity,
                        rate,
                        p,
                        dividend_yield,
                    ),
                }
            )
    return pd.DataFrame(rows)
