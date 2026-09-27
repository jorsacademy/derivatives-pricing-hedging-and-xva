"""Monte Carlo exposure profiles and transparent synthetic XVA measures."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .rates import (
    VasicekParams,
    par_swap_rate_vasicek,
    payer_swap_value_at_reset,
    simulate_vasicek_paths,
    vasicek_zcb_price,
)


@dataclass(frozen=True)
class XVAProblem:
    params: VasicekParams
    maturity_years: int = 7
    notional: float = 100_000_000.0
    n_paths: int = 20_000
    counterparty_hazard: float = 0.018
    own_hazard: float = 0.010
    counterparty_recovery: float = 0.40
    own_recovery: float = 0.40
    funding_spread: float = 0.006
    im_funding_spread: float = 0.0035
    pfe_quantile: float = 0.975
    im_multiplier: float = 0.55
    wrong_way_beta: float = 0.0
    seed: int = 17


@dataclass(frozen=True)
class XVAResult:
    exposure_profile: pd.DataFrame
    fixed_rate: float
    cva: float
    dva: float
    fva: float
    mva: float
    total_xva: float

    def to_dict(self) -> dict:
        return {
            "fixed_rate": round(self.fixed_rate, 8),
            "cva": round(self.cva, 2),
            "dva": round(self.dva, 2),
            "fva": round(self.fva, 2),
            "mva": round(self.mva, 2),
            "total_xva": round(self.total_xva, 2),
            "exposure_profile": self.exposure_profile.reset_index().to_dict(
                orient="records"
            ),
        }


def default_problem() -> XVAProblem:
    return XVAProblem(params=VasicekParams())


def _validate(problem: XVAProblem) -> None:
    if problem.maturity_years < 2:
        raise ValueError("maturity_years must be at least two")
    if problem.notional <= 0 or problem.n_paths < 100:
        raise ValueError("notional must be positive and n_paths at least 100")
    for hazard in (problem.counterparty_hazard, problem.own_hazard):
        if hazard < 0:
            raise ValueError("hazard rates must be nonnegative")
    for recovery in (problem.counterparty_recovery, problem.own_recovery):
        if not 0 <= recovery < 1:
            raise ValueError("recovery must lie in [0,1)")
    if not 0.5 < problem.pfe_quantile < 1.0:
        raise ValueError("pfe_quantile must lie in (0.5,1)")
    if problem.im_multiplier < 0:
        raise ValueError("im_multiplier must be nonnegative")


def _weighted_positive_exposure(
    values: np.ndarray,
    rates: np.ndarray,
    beta: float,
) -> float:
    positive = np.maximum(values, 0.0)
    if np.isclose(beta, 0.0) or np.isclose(np.std(rates), 0.0):
        return float(positive.mean())
    z = (rates - rates.mean()) / rates.std()
    weights = np.exp(np.clip(beta * z, -20.0, 20.0))
    weights /= weights.mean()
    return float(np.mean(positive * weights))


def solve(problem: XVAProblem | None = None) -> XVAResult:
    """Simulate a payer swap exposure profile and compute synthetic XVA."""
    p = problem or default_problem()
    _validate(p)

    times = np.arange(0, p.maturity_years + 1, dtype=float)
    payment_times = np.arange(1, p.maturity_years + 1, dtype=float)
    fixed_rate = par_swap_rate_vasicek(p.params, payment_times)
    rates = simulate_vasicek_paths(p.params, times, p.n_paths, p.seed)

    rows = []
    for i, time in enumerate(times):
        values = payer_swap_value_at_reset(
            rates[:, i],
            float(time),
            payment_times,
            fixed_rate,
            p.notional,
            p.params,
        )
        positive = np.maximum(values, 0.0)
        negative = np.maximum(-values, 0.0)
        epe = _weighted_positive_exposure(values, rates[:, i], p.wrong_way_beta)
        ene = float(negative.mean())
        pfe = float(np.quantile(positive, p.pfe_quantile))
        im = p.im_multiplier * max(pfe - epe, 0.0)
        rows.append(
            {
                "time": float(time),
                "expected_positive_exposure": epe,
                "expected_negative_exposure": ene,
                "pfe": pfe,
                "initial_margin_proxy": im,
            }
        )

    profile = pd.DataFrame(rows).set_index("time")

    cva = 0.0
    dva = 0.0
    fva = 0.0
    mva = 0.0
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        dt = t1 - t0
        mid = 0.5 * (t0 + t1)
        discount = float(
            vasicek_zcb_price(p.params.initial_rate, 0.0, mid, p.params)
        )

        epe = 0.5 * (
            profile.iloc[i - 1]["expected_positive_exposure"]
            + profile.iloc[i]["expected_positive_exposure"]
        )
        ene = 0.5 * (
            profile.iloc[i - 1]["expected_negative_exposure"]
            + profile.iloc[i]["expected_negative_exposure"]
        )
        im = 0.5 * (
            profile.iloc[i - 1]["initial_margin_proxy"]
            + profile.iloc[i]["initial_margin_proxy"]
        )

        cp_default_prob = np.exp(-p.counterparty_hazard * t0) - np.exp(
            -p.counterparty_hazard * t1
        )
        own_default_prob = np.exp(-p.own_hazard * t0) - np.exp(
            -p.own_hazard * t1
        )

        cva += (
            (1.0 - p.counterparty_recovery)
            * discount
            * epe
            * cp_default_prob
        )
        dva += (
            (1.0 - p.own_recovery)
            * discount
            * ene
            * own_default_prob
        )
        fva += discount * epe * p.funding_spread * dt
        mva += discount * im * p.im_funding_spread * dt

    total_xva = cva - dva + fva + mva
    return XVAResult(
        exposure_profile=profile,
        fixed_rate=fixed_rate,
        cva=float(cva),
        dva=float(dva),
        fva=float(fva),
        mva=float(mva),
        total_xva=float(total_xva),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
