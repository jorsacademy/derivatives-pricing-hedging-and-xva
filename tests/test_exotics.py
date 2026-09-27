import numpy as np

from derivatives_xva.exotics import (
    arithmetic_asian_call_mc,
    european_call_mc,
    up_and_out_call_mc,
)
from derivatives_xva.options import black_scholes_price


def test_european_mc_matches_black_scholes_within_sampling_error():
    mc = european_call_mc(
        100,
        100,
        1.0,
        0.03,
        0.20,
        n_paths=40_000,
        seed=31,
    )
    exact = black_scholes_price(100, 100, 1.0, 0.03, 0.20, "call")

    assert abs(mc.price - exact) <= 4.0 * mc.standard_error + 0.02


def test_barrier_call_is_cheaper_than_corresponding_european():
    barrier = up_and_out_call_mc(
        100,
        100,
        125,
        1.0,
        0.03,
        0.20,
        n_paths=25_000,
        seed=17,
    )
    european = european_call_mc(
        100,
        100,
        1.0,
        0.03,
        0.20,
        n_paths=25_000,
        seed=17,
    )
    asian = arithmetic_asian_call_mc(
        100,
        100,
        1.0,
        0.03,
        0.20,
        n_paths=25_000,
        seed=17,
    )

    assert 0 <= barrier.price < european.price
    assert asian.price > 0
