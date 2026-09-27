from dataclasses import replace

import numpy as np

from derivatives_xva.xva import default_problem as default_xva_problem
from derivatives_xva.xva_hedge import (
    benchmark_max_normalized_residual,
    default_problem,
    estimate_xva_sensitivities,
    solve,
)


def test_xva_finite_difference_sensitivities_are_finite():
    xva = replace(default_xva_problem(), n_paths=1_500, seed=41)
    sensitivities = estimate_xva_sensitivities(xva)
    assert np.isfinite(sensitivities.to_numpy()).all()
    assert len(sensitivities) == 4


def test_xva_hedge_reduces_max_normalized_residual():
    xva = replace(default_xva_problem(), n_paths=1_500, seed=43)
    problem = default_problem(xva)
    result = solve(problem)

    assert result.max_normalized_residual < benchmark_max_normalized_residual(problem)
    assert result.gross_notional <= problem.gross_notional_limit + 1e-8
