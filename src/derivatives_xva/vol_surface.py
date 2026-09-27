"""SVI and SABR volatility-surface calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def validate(self) -> None:
        if self.a < 0 or self.b <= 0 or self.sigma <= 0:
            raise ValueError("SVI requires a>=0, b>0, sigma>0")
        if not -1 < self.rho < 1:
            raise ValueError("SVI rho must lie in (-1,1)")


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float

    def validate(self) -> None:
        if self.alpha <= 0 or self.nu < 0:
            raise ValueError("SABR requires alpha>0 and nu>=0")
        if not 0 <= self.beta <= 1:
            raise ValueError("SABR beta must lie in [0,1]")
        if not -1 < self.rho < 1:
            raise ValueError("SABR rho must lie in (-1,1)")


def svi_total_variance(
    log_moneyness: np.ndarray | float,
    params: SVIParams,
) -> np.ndarray:
    """Raw-SVI total variance w(k)."""
    params.validate()
    k = np.asarray(log_moneyness, dtype=float)
    x = k - params.m
    return params.a + params.b * (
        params.rho * x + np.sqrt(x * x + params.sigma * params.sigma)
    )


def calibrate_svi(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
) -> tuple[SVIParams, float]:
    """Calibrate raw SVI by bounded nonlinear least squares."""
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.ndim != 1 or w.ndim != 1 or len(k) != len(w) or len(k) < 5:
        raise ValueError("need at least five aligned SVI observations")
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("total variance observations must be positive and finite")

    x0 = np.array([max(float(w.min()) * 0.5, 1e-6), 0.10, 0.0, 0.0, 0.20])
    lower = np.array([0.0, 1e-8, -0.999, -2.0, 1e-4])
    upper = np.array([5.0, 5.0, 0.999, 2.0, 5.0])

    def residual(x: np.ndarray) -> np.ndarray:
        p = SVIParams(*map(float, x))
        return svi_total_variance(k, p) - w

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"SVI calibration failed: {result.message}")
    params = SVIParams(*map(float, result.x))
    rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return params, rmse


def sabr_implied_vol(
    forward: float,
    strike: float,
    maturity: float,
    params: SABRParams,
) -> float:
    """Hagan lognormal SABR implied-volatility approximation."""
    params.validate()
    if forward <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("forward, strike and maturity must be positive")

    alpha, beta, rho, nu = (
        params.alpha,
        params.beta,
        params.rho,
        params.nu,
    )
    one_minus_beta = 1.0 - beta

    if np.isclose(forward, strike):
        f_beta = forward ** one_minus_beta
        correction = (
            (one_minus_beta**2 / 24.0) * alpha**2 / forward ** (2.0 * one_minus_beta)
            + rho * beta * nu * alpha / (4.0 * f_beta)
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        )
        return float(alpha / f_beta * (1.0 + correction * maturity))

    log_fk = np.log(forward / strike)
    fk_beta = (forward * strike) ** (one_minus_beta / 2.0)
    z = (nu / alpha) * fk_beta * log_fk
    root = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = np.log((root + z - rho) / (1.0 - rho))
    z_over_x = 1.0 if abs(z) < 1e-12 else z / x_z

    denominator = fk_beta * (
        1.0
        + one_minus_beta**2 * log_fk**2 / 24.0
        + one_minus_beta**4 * log_fk**4 / 1920.0
    )
    correction = (
        one_minus_beta**2 * alpha**2 / (24.0 * fk_beta**2)
        + rho * beta * nu * alpha / (4.0 * fk_beta)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )
    return float(alpha / denominator * z_over_x * (1.0 + correction * maturity))


def calibrate_sabr(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    forward: float,
    maturity: float,
    beta: float = 0.5,
) -> tuple[SABRParams, float]:
    """Calibrate alpha, rho and nu with beta held fixed."""
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    if (
        strikes.ndim != 1
        or market_vols.ndim != 1
        or len(strikes) != len(market_vols)
        or len(strikes) < 3
    ):
        raise ValueError("need at least three aligned SABR observations")
    if (strikes <= 0).any() or (market_vols <= 0).any():
        raise ValueError("strikes and market vols must be positive")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")

    def residual(x: np.ndarray) -> np.ndarray:
        params = SABRParams(float(x[0]), beta, float(x[1]), float(x[2]))
        return np.array(
            [
                sabr_implied_vol(forward, float(k), maturity, params)
                for k in strikes
            ]
        ) - market_vols

    result = least_squares(
        residual,
        x0=np.array([0.20, 0.0, 0.50]),
        bounds=(
            np.array([1e-4, -0.999, 1e-4]),
            np.array([5.0, 0.999, 5.0]),
        ),
        max_nfev=3000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"SABR calibration failed: {result.message}")
    params = SABRParams(float(result.x[0]), beta, float(result.x[1]), float(result.x[2]))
    rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return params, rmse
