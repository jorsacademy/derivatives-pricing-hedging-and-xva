import numpy as np

from derivatives_xva.vol_surface import (
    SABRParams,
    SVIParams,
    calibrate_sabr,
    calibrate_svi,
    sabr_implied_vol,
    svi_total_variance,
)


def test_svi_calibration_recovers_synthetic_smile():
    true = SVIParams(a=0.02, b=0.18, rho=-0.35, m=-0.05, sigma=0.22)
    k = np.linspace(-0.4, 0.4, 13)
    w = svi_total_variance(k, true)
    fitted, rmse = calibrate_svi(k, w)

    assert rmse < 1e-8
    assert np.allclose(svi_total_variance(k, fitted), w, atol=1e-7)


def test_sabr_calibration_recovers_synthetic_vols():
    true = SABRParams(alpha=0.25, beta=0.5, rho=-0.30, nu=0.70)
    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    market = np.array(
        [sabr_implied_vol(100.0, k, 2.0, true) for k in strikes]
    )
    fitted, rmse = calibrate_sabr(strikes, market, 100.0, 2.0, beta=0.5)

    assert rmse < 1e-8
    assert np.allclose(
        [sabr_implied_vol(100.0, k, 2.0, fitted) for k in strikes],
        market,
        atol=1e-7,
    )
