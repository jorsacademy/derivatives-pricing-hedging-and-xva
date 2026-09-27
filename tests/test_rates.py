import numpy as np

from derivatives_xva.rates import (
    VasicekParams,
    par_swap_rate_vasicek,
    payer_swap_value_at_reset,
    simulate_vasicek_paths,
    vasicek_zcb_price,
)


def test_vasicek_zero_coupon_prices_are_positive():
    p = VasicekParams()
    prices = np.array([vasicek_zcb_price(p.initial_rate, 0.0, t, p) for t in [1, 3, 5, 10]])
    assert (prices > 0).all()
    assert (np.diff(prices) < 0).all()


def test_par_swap_has_zero_initial_value():
    p = VasicekParams()
    payment_times = np.arange(1, 8, dtype=float)
    fixed = par_swap_rate_vasicek(p, payment_times)
    value = payer_swap_value_at_reset(
        p.initial_rate,
        0.0,
        payment_times,
        fixed,
        100_000_000.0,
        p,
    )
    assert abs(float(value)) < 1e-6


def test_vasicek_simulation_is_reproducible():
    p = VasicekParams()
    times = np.arange(0, 5, dtype=float)
    a = simulate_vasicek_paths(p, times, n_paths=1000, seed=99)
    b = simulate_vasicek_paths(p, times, n_paths=1000, seed=99)
    assert np.allclose(a, b)
