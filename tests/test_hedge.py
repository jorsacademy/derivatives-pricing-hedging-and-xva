import numpy as np

from derivatives_xva.hedge import default_problem, solve


def test_robust_hedge_reduces_worst_stress_loss():
    p = default_problem()
    r = solve(p)
    assert r.worst_loss < r.unhedged_worst_loss


def test_robust_hedge_respects_limits_and_reconciles_pnl():
    p = default_problem()
    r = solve(p)

    assert r.gross_notional <= p.gross_notional_limit + 1e-9
    for hedge, notional in r.hedge_notionals.items():
        assert abs(notional) <= p.maximum_notional.loc[hedge] + 1e-9

    recomputed = p.portfolio_stress_pnl + p.hedge_stress_pnl @ r.hedge_notionals
    assert np.allclose(recomputed.to_numpy(), r.residual_stress_pnl.to_numpy())
