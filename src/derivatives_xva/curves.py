"""Yield-curve utilities for synthetic derivatives research examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve with linear zero-rate interpolation."""

    maturities: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=float)
        rates = np.asarray(self.zero_rates, dtype=float)
        if maturities.ndim != 1 or rates.ndim != 1:
            raise ValueError("maturities and zero_rates must be one-dimensional")
        if len(maturities) != len(rates) or len(maturities) == 0:
            raise ValueError("maturities and zero_rates must have equal nonzero length")
        if (maturities <= 0).any() or not np.all(np.diff(maturities) > 0):
            raise ValueError("maturities must be positive and strictly increasing")
        if not np.isfinite(rates).all():
            raise ValueError("zero rates must be finite")
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "zero_rates", rates)

    def zero_rate(self, maturity: float | np.ndarray) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)
        if (t < 0).any():
            raise ValueError("maturity must be nonnegative")
        return np.interp(
            t,
            self.maturities,
            self.zero_rates,
            left=self.zero_rates[0],
            right=self.zero_rates[-1],
        )

    def discount(self, maturity: float | np.ndarray) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, start: float, end: float) -> float:
        if start < 0 or end <= start:
            raise ValueError("require 0 <= start < end")
        p0 = 1.0 if np.isclose(start, 0.0) else float(self.discount(start))
        p1 = float(self.discount(end))
        return float(np.log(p0 / p1) / (end - start))


def bootstrap_par_swap_curve(
    maturities: np.ndarray,
    par_rates: np.ndarray,
) -> ZeroCurve:
    """Bootstrap annual-pay discount factors from consecutive par-swap rates.

    The educational bootstrap assumes maturities are 1, 2, ..., N years and
    coupons are paid annually. Rates are decimal values, e.g. 0.04 for 4%.
    """
    maturities = np.asarray(maturities, dtype=float)
    par_rates = np.asarray(par_rates, dtype=float)
    if len(maturities) != len(par_rates) or len(maturities) == 0:
        raise ValueError("maturities and par_rates must have equal nonzero length")
    expected = np.arange(1, len(maturities) + 1, dtype=float)
    if not np.allclose(maturities, expected):
        raise ValueError("educational bootstrap requires consecutive annual maturities")
    if (par_rates <= -1.0).any() or not np.isfinite(par_rates).all():
        raise ValueError("invalid par rates")

    discounts: list[float] = []
    for rate in par_rates:
        previous_coupon_pv = float(rate * np.sum(discounts))
        discount = (1.0 - previous_coupon_pv) / (1.0 + rate)
        if discount <= 0:
            raise ValueError("par rates imply a nonpositive discount factor")
        discounts.append(discount)

    discounts_array = np.asarray(discounts, dtype=float)
    zero_rates = -np.log(discounts_array) / maturities
    return ZeroCurve(maturities=maturities, zero_rates=zero_rates)


def par_swap_rate(curve: ZeroCurve, maturity_years: int) -> float:
    """Return annual-pay par swap rate for an integer maturity."""
    if maturity_years < 1:
        raise ValueError("maturity_years must be at least one")
    payment_times = np.arange(1, maturity_years + 1, dtype=float)
    discounts = curve.discount(payment_times)
    return float((1.0 - discounts[-1]) / discounts.sum())
