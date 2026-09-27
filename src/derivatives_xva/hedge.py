"""Robust stress-hedge optimization for a synthetic derivatives book."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class HedgeProblem:
    portfolio_stress_pnl: pd.Series
    hedge_stress_pnl: pd.DataFrame
    hedge_cost_per_unit: pd.Series
    maximum_notional: pd.Series
    gross_notional_limit: float = 1.5
    cost_weight: float = 0.35


@dataclass(frozen=True)
class HedgeResult:
    hedge_notionals: pd.Series
    residual_stress_pnl: pd.Series
    worst_loss: float
    unhedged_worst_loss: float
    hedge_cost: float
    gross_notional: float
    objective_value: float

    @property
    def worst_loss_reduction(self) -> float:
        return self.unhedged_worst_loss - self.worst_loss

    def to_dict(self) -> dict:
        return {
            "hedge_notionals": self.hedge_notionals.round(6).to_dict(),
            "residual_stress_pnl": self.residual_stress_pnl.round(6).to_dict(),
            "worst_loss": round(self.worst_loss, 6),
            "unhedged_worst_loss": round(self.unhedged_worst_loss, 6),
            "worst_loss_reduction": round(self.worst_loss_reduction, 6),
            "hedge_cost": round(self.hedge_cost, 6),
            "gross_notional": round(self.gross_notional, 6),
            "objective_value": round(self.objective_value, 6),
        }


def default_problem() -> HedgeProblem:
    scenarios = [
        "equity_crash_vol_spike",
        "equity_selloff",
        "vol_spike_flat_spot",
        "moderate_rally",
        "melt_up_vol_crush",
    ]
    portfolio = pd.Series(
        [-42.0, -23.0, -12.0, 8.0, 15.0],
        index=scenarios,
        name="portfolio_pnl_mm",
    )
    hedges = pd.DataFrame(
        {
            "put_spread": [28.0, 16.0, 5.0, -3.0, -5.0],
            "vega_overlay": [12.0, 8.0, 10.0, -3.0, -6.0],
            "short_futures": [17.0, 10.0, 0.0, -10.0, -18.0],
        },
        index=scenarios,
    )
    return HedgeProblem(
        portfolio_stress_pnl=portfolio,
        hedge_stress_pnl=hedges,
        hedge_cost_per_unit=pd.Series(
            {"put_spread": 1.2, "vega_overlay": 0.8, "short_futures": 0.35}
        ),
        maximum_notional=pd.Series(
            {"put_spread": 2.0, "vega_overlay": 2.0, "short_futures": 2.0}
        ),
    )


def _validate(problem: HedgeProblem) -> None:
    scenarios = list(problem.portfolio_stress_pnl.index)
    hedges = list(problem.hedge_stress_pnl.columns)
    if list(problem.hedge_stress_pnl.index) != scenarios:
        raise ValueError("stress scenarios must align")
    if list(problem.hedge_cost_per_unit.index) != hedges:
        raise ValueError("hedge costs must align with hedge columns")
    if list(problem.maximum_notional.index) != hedges:
        raise ValueError("notional limits must align with hedge columns")
    if problem.gross_notional_limit <= 0 or problem.cost_weight < 0:
        raise ValueError("invalid gross limit or cost weight")
    if (problem.maximum_notional < 0).any():
        raise ValueError("notional limits must be nonnegative")


def solve(problem: HedgeProblem | None = None) -> HedgeResult:
    """Minimize worst stress loss plus a linear hedge-cost penalty."""
    p = problem or default_problem()
    _validate(p)

    scenarios = list(p.portfolio_stress_pnl.index)
    hedges = list(p.hedge_stress_pnl.columns)
    n_h = len(hedges)
    idx_z = 2 * n_h
    n_vars = idx_z + 1

    c = np.zeros(n_vars)
    costs = p.hedge_cost_per_unit.loc[hedges].to_numpy(dtype=float)
    c[:n_h] = p.cost_weight * costs
    c[n_h : 2 * n_h] = p.cost_weight * costs
    c[idx_z] = 1.0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for scenario in scenarios:
        h = p.hedge_stress_pnl.loc[scenario, hedges].to_numpy(dtype=float)
        portfolio_pnl = float(p.portfolio_stress_pnl.loc[scenario])
        row = np.zeros(n_vars)
        row[:n_h] = -h
        row[n_h : 2 * n_h] = h
        row[idx_z] = -1.0
        a_ub.append(row)
        b_ub.append(portfolio_pnl)

    gross = np.zeros(n_vars)
    gross[: 2 * n_h] = 1.0
    a_ub.append(gross)
    b_ub.append(float(p.gross_notional_limit))

    bounds: list[tuple[float | None, float | None]] = []
    for hedge in hedges:
        bounds.append((0.0, float(p.maximum_notional.loc[hedge])))
    for hedge in hedges:
        bounds.append((0.0, float(p.maximum_notional.loc[hedge])))
    bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"hedge optimization failed: {result.message}")

    notionals = result.x[:n_h] - result.x[n_h : 2 * n_h]
    hedge_notionals = pd.Series(notionals, index=hedges, name="notional")
    residual = p.portfolio_stress_pnl + p.hedge_stress_pnl @ hedge_notionals
    worst_loss = float(np.maximum(-residual.to_numpy(dtype=float), 0.0).max())
    unhedged = float(
        np.maximum(-p.portfolio_stress_pnl.to_numpy(dtype=float), 0.0).max()
    )
    hedge_cost = float((hedge_notionals.abs() * p.hedge_cost_per_unit).sum())

    return HedgeResult(
        hedge_notionals=hedge_notionals,
        residual_stress_pnl=residual,
        worst_loss=worst_loss,
        unhedged_worst_loss=unhedged,
        hedge_cost=hedge_cost,
        gross_notional=float(hedge_notionals.abs().sum()),
        objective_value=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
