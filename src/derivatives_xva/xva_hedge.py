"""Robust optimization of an XVA-sensitivity hedge portfolio."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .rates import VasicekParams
from .xva import XVAProblem, default_problem as default_xva_problem, solve as solve_xva


@dataclass(frozen=True)
class XVAHedgeProblem:
    base_sensitivities: pd.Series
    hedge_sensitivities: pd.DataFrame
    hedge_cost_per_unit: pd.Series
    maximum_notional: pd.Series
    gross_notional_limit: float = 2.0
    cost_weight: float = 0.08


@dataclass(frozen=True)
class XVAHedgeResult:
    hedge_notionals: pd.Series
    residual_sensitivities: pd.Series
    max_normalized_residual: float
    hedge_cost: float
    gross_notional: float
    objective_value: float

def estimate_xva_sensitivities(
    problem: XVAProblem | None = None,
    bump: float = 1e-4,
) -> pd.Series:
    """Finite-difference total-XVA sensitivities using common random numbers."""
    if bump <= 0:
        raise ValueError("bump must be positive")
    p = problem or default_xva_problem()
    base = solve_xva(p).total_xva

    hazard = solve_xva(
        replace(p, counterparty_hazard=p.counterparty_hazard + bump)
    ).total_xva
    funding = solve_xva(
        replace(p, funding_spread=p.funding_spread + bump)
    ).total_xva

    rate_params = replace(p.params, initial_rate=p.params.initial_rate + bump)
    rate = solve_xva(replace(p, params=rate_params)).total_xva

    vol_params = replace(p.params, volatility=p.params.volatility + bump)
    vol = solve_xva(replace(p, params=vol_params)).total_xva

    return pd.Series(
        {
            "counterparty_hazard_1bp": hazard - base,
            "funding_spread_1bp": funding - base,
            "initial_rate_1bp": rate - base,
            "rate_volatility_1bp": vol - base,
        },
        name="xva_sensitivity",
    )


def default_problem(
    xva_problem: XVAProblem | None = None,
) -> XVAHedgeProblem:
    """Build a synthetic XVA hedge problem from finite-difference sensitivities."""
    p = xva_problem or replace(default_xva_problem(), n_paths=5_000)
    base = estimate_xva_sensitivities(p)
    b = base.to_numpy(dtype=float)

    # Synthetic hedge effectiveness is expressed relative to the measured XVA
    # sensitivities so scale remains meaningful as the exposure engine changes.
    matrix = pd.DataFrame(
        {
            "counterparty_cds": [-0.85 * b[0], -0.03 * b[1], 0.0, 0.0],
            "funding_basis_swap": [0.0, -0.80 * b[1], -0.05 * b[2], 0.0],
            "rates_swap_overlay": [0.0, 0.0, -0.75 * b[2], -0.15 * b[3]],
            "swaption_overlay": [0.0, 0.0, -0.10 * b[2], -0.80 * b[3]],
        },
        index=base.index,
    )
    hedges = matrix.columns
    return XVAHedgeProblem(
        base_sensitivities=base,
        hedge_sensitivities=matrix,
        hedge_cost_per_unit=pd.Series(
            [0.18, 0.12, 0.08, 0.22],
            index=hedges,
            name="cost",
        ),
        maximum_notional=pd.Series(
            [1.5, 1.5, 1.5, 1.5],
            index=hedges,
            name="max_notional",
        ),
    )


def _validate(problem: XVAHedgeProblem) -> None:
    factors = list(problem.base_sensitivities.index)
    hedges = list(problem.hedge_sensitivities.columns)
    if list(problem.hedge_sensitivities.index) != factors:
        raise ValueError("hedge sensitivities must align with base factors")
    if list(problem.hedge_cost_per_unit.index) != hedges:
        raise ValueError("hedge costs must align with hedge columns")
    if list(problem.maximum_notional.index) != hedges:
        raise ValueError("notional limits must align with hedge columns")
    if problem.gross_notional_limit <= 0 or problem.cost_weight < 0:
        raise ValueError("invalid hedge budget or cost weight")


def benchmark_max_normalized_residual(problem: XVAHedgeProblem) -> float:
    scale = np.maximum(
        np.abs(problem.base_sensitivities.to_numpy(dtype=float)),
        1.0,
    )
    return float(
        np.max(
            np.abs(problem.base_sensitivities.to_numpy(dtype=float)) / scale
        )
    )


def solve(problem: XVAHedgeProblem | None = None) -> XVAHedgeResult:
    """Minimize maximum normalized residual XVA sensitivity plus hedge cost."""
    p = problem or default_problem()
    _validate(p)

    factors = list(p.base_sensitivities.index)
    hedges = list(p.hedge_sensitivities.columns)
    n_h = len(hedges)
    idx_z = 2 * n_h
    n_vars = idx_z + 1

    base = p.base_sensitivities.loc[factors].to_numpy(dtype=float)
    H = p.hedge_sensitivities.loc[factors, hedges].to_numpy(dtype=float)
    scale = np.maximum(np.abs(base), 1.0)

    c = np.zeros(n_vars)
    costs = p.hedge_cost_per_unit.loc[hedges].to_numpy(dtype=float)
    c[:n_h] = p.cost_weight * costs
    c[n_h:2 * n_h] = p.cost_weight * costs
    c[idx_z] = 1.0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for i in range(len(factors)):
        # +(base_i + H_i x)/scale_i <= z
        row = np.zeros(n_vars)
        row[:n_h] = H[i] / scale[i]
        row[n_h:2 * n_h] = -H[i] / scale[i]
        row[idx_z] = -1.0
        a_ub.append(row)
        b_ub.append(-base[i] / scale[i])

        # -(base_i + H_i x)/scale_i <= z
        row = np.zeros(n_vars)
        row[:n_h] = -H[i] / scale[i]
        row[n_h:2 * n_h] = H[i] / scale[i]
        row[idx_z] = -1.0
        a_ub.append(row)
        b_ub.append(base[i] / scale[i])

    gross = np.zeros(n_vars)
    gross[:2 * n_h] = 1.0
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
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"XVA hedge optimization failed: {result.message}")

    signed = result.x[:n_h] - result.x[n_h:2 * n_h]
    notionals = pd.Series(signed, index=hedges, name="hedge_notional")
    residual = p.base_sensitivities + p.hedge_sensitivities @ notionals
    normalized = np.abs(residual.to_numpy(dtype=float)) / scale
    hedge_cost = float((notionals.abs() * p.hedge_cost_per_unit).sum())

    return XVAHedgeResult(
        hedge_notionals=notionals,
        residual_sensitivities=residual,
        max_normalized_residual=float(normalized.max()),
        hedge_cost=hedge_cost,
        gross_notional=float(notionals.abs().sum()),
        objective_value=float(result.fun),
    )
