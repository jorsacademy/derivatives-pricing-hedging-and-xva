import numpy as np

from derivatives_xva.heston import (
    HestonParams,
    calibrate_heston,
    heston_call_price,
    synthetic_heston_market,
)
from derivatives_xva.options import black_scholes_price


def test_heston_collapses_toward_black_scholes_for_nearly_constant_variance():
    p = HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=1e-4,
        rho=0.0,
        v0=0.04,
    )
    heston = heston_call_price(100, 100, 1.0, 0.03, p)
    bs = black_scholes_price(100, 100, 1.0, 0.03, 0.20, "call")
    assert np.isclose(heston, bs, atol=2e-3)


def test_heston_calibration_reprices_synthetic_surface():
    market = synthetic_heston_market()
    result = calibrate_heston(market, spot=100.0, rate=0.03)

    assert result.rmse < 2e-3
    assert result.params.rho < 0
    assert np.isfinite(result.fitted["model_price"]).all()
