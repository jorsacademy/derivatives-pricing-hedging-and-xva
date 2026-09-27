import numpy as np

from derivatives_xva.curves import bootstrap_par_swap_curve, par_swap_rate


def test_bootstrap_reprices_input_par_swaps():
    maturities = np.arange(1, 6, dtype=float)
    rates = np.array([0.032, 0.034, 0.036, 0.0375, 0.039])
    curve = bootstrap_par_swap_curve(maturities, rates)

    for maturity, expected in zip(maturities.astype(int), rates):
        assert np.isclose(par_swap_rate(curve, maturity), expected, atol=1e-12)


def test_discount_factors_are_positive_and_declining_for_positive_curve():
    curve = bootstrap_par_swap_curve(
        np.arange(1, 5, dtype=float),
        np.array([0.03, 0.032, 0.034, 0.036]),
    )
    discounts = curve.discount(np.arange(1, 5, dtype=float))
    assert (discounts > 0).all()
    assert (np.diff(discounts) < 0).all()
