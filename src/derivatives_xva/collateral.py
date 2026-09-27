"""Netting-set aggregation and stylized CSA collateral mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CSATerms:
    threshold_receive: float = 500_000.0
    threshold_post: float = 500_000.0
    minimum_transfer_amount: float = 100_000.0
    margin_period_steps: int = 1

    def validate(self) -> None:
        if self.threshold_receive < 0 or self.threshold_post < 0:
            raise ValueError("CSA thresholds must be nonnegative")
        if self.minimum_transfer_amount < 0:
            raise ValueError("minimum transfer amount must be nonnegative")
        if self.margin_period_steps < 0:
            raise ValueError("margin_period_steps must be nonnegative")


@dataclass(frozen=True)
class CollateralExposureResult:
    exposure_profile: pd.DataFrame
    net_mtm: np.ndarray
    collateral_balance: np.ndarray
    unsecured_mtm: np.ndarray

    @property
    def mean_epe(self) -> float:
        return float(self.exposure_profile["epe"].mean())


def synthetic_netting_set_paths(
    times: np.ndarray,
    n_paths: int = 10_000,
    seed: int = 11,
) -> np.ndarray:
    """Generate three correlated synthetic trade MtM paths.

    Output shape is paths x times x trades. Currency scale is approximately
    millions and all trades start at zero MtM.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isclose(times[0], 0.0):
        raise ValueError("times must be a one-dimensional grid starting at zero")
    if not np.all(np.diff(times) > 0) or n_paths < 100:
        raise ValueError("times must increase and n_paths must be at least 100")

    rng = np.random.default_rng(seed)
    n_times = len(times)
    f1 = np.zeros((n_paths, n_times), dtype=float)
    f2 = np.zeros_like(f1)

    for i, dt in enumerate(np.diff(times), start=1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        f1[:, i] = 0.75 * f1[:, i - 1] + np.sqrt(dt) * z1
        f2[:, i] = (
            0.65 * f2[:, i - 1]
            + np.sqrt(dt) * (0.35 * z1 + np.sqrt(1.0 - 0.35**2) * z2)
        )

    trade_1 = 4.0e6 * f1 + 0.30e6 * times
    trade_2 = -2.8e6 * f1 + 2.2e6 * f2 - 0.15e6 * times
    trade_3 = 1.0e6 * f1 - 1.2e6 * f2
    return np.stack([trade_1, trade_2, trade_3], axis=-1)


def collateralize_netting_set(
    trade_mtm_paths: np.ndarray,
    times: np.ndarray,
    terms: CSATerms,
    pfe_quantile: float = 0.975,
) -> CollateralExposureResult:
    """Apply bilateral variation-margin logic to a netting set.

    Positive collateral balance means collateral held by the dealer; negative
    balance means collateral posted. Margin calls are based on lagged net MtM
    according to margin_period_steps and are ignored below the MTA.
    """
    terms.validate()
    paths = np.asarray(trade_mtm_paths, dtype=float)
    times = np.asarray(times, dtype=float)
    if paths.ndim != 3:
        raise ValueError("trade_mtm_paths must have shape paths x times x trades")
    if paths.shape[1] != len(times):
        raise ValueError("time dimension must align with times")
    if not 0.5 < pfe_quantile < 1.0:
        raise ValueError("pfe_quantile must lie in (0.5,1)")
    if not np.isfinite(paths).all():
        raise ValueError("trade MtM paths must be finite")

    net = paths.sum(axis=-1)
    n_paths, n_times = net.shape
    collateral = np.zeros_like(net)

    for t in range(1, n_times):
        collateral[:, t] = collateral[:, t - 1]
        source = max(t - terms.margin_period_steps, 0)
        reference_mtm = net[:, source]
        target = np.where(
            reference_mtm > terms.threshold_receive,
            reference_mtm - terms.threshold_receive,
            np.where(
                reference_mtm < -terms.threshold_post,
                reference_mtm + terms.threshold_post,
                0.0,
            ),
        )
        transfer = target - collateral[:, t - 1]
        execute = np.abs(transfer) >= terms.minimum_transfer_amount
        collateral[execute, t] = target[execute]

    unsecured = net - collateral
    positive = np.maximum(unsecured, 0.0)
    negative = np.maximum(-unsecured, 0.0)

    profile = pd.DataFrame(
        {
            "epe": positive.mean(axis=0),
            "ene": negative.mean(axis=0),
            "pfe": np.quantile(positive, pfe_quantile, axis=0),
            "mean_abs_collateral": np.abs(collateral).mean(axis=0),
        },
        index=pd.Index(times, name="time"),
    )
    return CollateralExposureResult(
        exposure_profile=profile,
        net_mtm=net,
        collateral_balance=collateral,
        unsecured_mtm=unsecured,
    )


def uncollateralized_profile(
    trade_mtm_paths: np.ndarray,
    times: np.ndarray,
    pfe_quantile: float = 0.975,
) -> pd.DataFrame:
    paths = np.asarray(trade_mtm_paths, dtype=float)
    net = paths.sum(axis=-1)
    positive = np.maximum(net, 0.0)
    negative = np.maximum(-net, 0.0)
    return pd.DataFrame(
        {
            "epe": positive.mean(axis=0),
            "ene": negative.mean(axis=0),
            "pfe": np.quantile(positive, pfe_quantile, axis=0),
        },
        index=pd.Index(np.asarray(times, dtype=float), name="time"),
    )
