from dataclasses import replace

import numpy as np

from derivatives_xva.xva import default_problem, solve


def test_xva_engine_returns_finite_positive_adjustments():
    p = replace(default_problem(), n_paths=4000)
    r = solve(p)

    assert r.cva > 0
    assert r.fva > 0
    assert r.mva >= 0
    assert np.isfinite(r.total_xva)
    assert np.isclose(r.exposure_profile.iloc[-1]["pfe"], 0.0)


def test_wrong_way_proxy_increases_cva_for_payer_swap_rate_exposure():
    base = replace(default_problem(), n_paths=5000, wrong_way_beta=0.0, seed=23)
    wwr = replace(base, wrong_way_beta=0.75)
    r0 = solve(base)
    r1 = solve(wwr)
    assert r1.cva >= r0.cva
