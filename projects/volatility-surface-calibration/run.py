from pathlib import Path

import numpy as np
import pandas as pd

from derivatives_xva.vol_surface import (
    SABRParams,
    SVIParams,
    calibrate_sabr,
    calibrate_svi,
    sabr_implied_vol,
    svi_total_variance,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    true_svi = SVIParams(a=0.018, b=0.16, rho=-0.42, m=-0.04, sigma=0.24)
    k = np.linspace(-0.45, 0.45, 17)
    market_w = svi_total_variance(k, true_svi)
    fitted_svi, svi_rmse = calibrate_svi(k, market_w)

    svi_table = pd.DataFrame(
        {
            "log_moneyness": k,
            "market_total_variance": market_w,
            "model_total_variance": svi_total_variance(k, fitted_svi),
        }
    )
    svi_table["error"] = (
        svi_table["model_total_variance"] - svi_table["market_total_variance"]
    )
    svi_table.to_csv(output_dir / "svi_fit.csv", index=False)

    forward = 100.0
    maturity = 2.0
    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    true_sabr = SABRParams(alpha=0.24, beta=0.5, rho=-0.28, nu=0.72)
    market_vols = np.array(
        [sabr_implied_vol(forward, strike, maturity, true_sabr) for strike in strikes]
    )
    fitted_sabr, sabr_rmse = calibrate_sabr(
        strikes,
        market_vols,
        forward,
        maturity,
        beta=0.5,
    )

    sabr_table = pd.DataFrame(
        {
            "strike": strikes,
            "market_vol": market_vols,
            "model_vol": [
                sabr_implied_vol(forward, strike, maturity, fitted_sabr)
                for strike in strikes
            ],
        }
    )
    sabr_table["error"] = sabr_table["model_vol"] - sabr_table["market_vol"]
    sabr_table.to_csv(output_dir / "sabr_fit.csv", index=False)

    pd.DataFrame(
        [
            {"model": "SVI", "rmse": svi_rmse},
            {"model": "SABR", "rmse": sabr_rmse},
        ]
    ).to_csv(output_dir / "calibration_summary.csv", index=False)

    print("SVI", fitted_svi, "RMSE", svi_rmse)
    print("SABR", fitted_sabr, "RMSE", sabr_rmse)


if __name__ == "__main__":
    main()
