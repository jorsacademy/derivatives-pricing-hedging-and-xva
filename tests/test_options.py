import numpy as np

from derivatives_xva.options import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)


def test_put_call_parity():
    s, k, t, r, vol, q = 100.0, 105.0, 1.4, 0.03, 0.24, 0.01
    call = black_scholes_price(s, k, t, r, vol, "call", q)
    put = black_scholes_price(s, k, t, r, vol, "put", q)
    lhs = call - put
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert np.isclose(lhs, rhs, atol=1e-10)


def test_implied_volatility_recovers_input_volatility():
    price = black_scholes_price(100, 100, 2.0, 0.025, 0.31, "call")
    recovered = implied_volatility(price, 100, 100, 2.0, 0.025, "call")
    assert np.isclose(recovered, 0.31, atol=1e-9)


def test_greeks_have_expected_signs():
    call = black_scholes_greeks(100, 100, 1.0, 0.03, 0.2, "call")
    put = black_scholes_greeks(100, 100, 1.0, 0.03, 0.2, "put")
    assert 0 < call.delta < 1
    assert -1 < put.delta < 0
    assert call.gamma > 0 and call.vega > 0
