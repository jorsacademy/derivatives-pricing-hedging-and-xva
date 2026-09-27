import numpy as np

from derivatives_xva.collateral import (
    CSATerms,
    collateralize_netting_set,
    synthetic_netting_set_paths,
    uncollateralized_profile,
)


def test_csa_collateral_reduces_average_positive_exposure():
    times = np.arange(0, 8, dtype=float)
    paths = synthetic_netting_set_paths(times, n_paths=5_000, seed=19)
    uncollateralized = uncollateralized_profile(paths, times)
    collateralized = collateralize_netting_set(paths, times, CSATerms())

    assert collateralized.mean_epe < float(uncollateralized["epe"].mean())
    assert (
        collateralized.exposure_profile["mean_abs_collateral"].iloc[-1] > 0
    )


def test_netting_set_shapes_and_reconciliation():
    times = np.arange(0, 5, dtype=float)
    paths = synthetic_netting_set_paths(times, n_paths=1_000, seed=5)
    result = collateralize_netting_set(paths, times, CSATerms())

    assert result.net_mtm.shape == (1_000, len(times))
    assert np.allclose(
        result.unsecured_mtm,
        result.net_mtm - result.collateral_balance,
    )
