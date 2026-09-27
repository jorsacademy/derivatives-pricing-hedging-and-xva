from pathlib import Path

import pandas as pd

from derivatives_xva.heston import calibrate_heston, synthetic_heston_market


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    market = synthetic_heston_market()
    result = calibrate_heston(market, spot=100.0, rate=0.03)

    market.to_csv(output_dir / "synthetic_option_surface.csv", index=False)
    result.fitted.to_csv(output_dir / "heston_fitted_surface.csv", index=False)
    pd.DataFrame(
        [
            {
                "kappa": result.params.kappa,
                "theta": result.params.theta,
                "sigma": result.params.sigma,
                "rho": result.params.rho,
                "v0": result.params.v0,
                "rmse": result.rmse,
            }
        ]
    ).to_csv(output_dir / "heston_parameters.csv", index=False)

    print(result.params)
    print("RMSE", result.rmse)


if __name__ == "__main__":
    main()
