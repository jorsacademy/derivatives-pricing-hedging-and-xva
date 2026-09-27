from pathlib import Path

import numpy as np
import pandas as pd

from derivatives_xva.collateral import (
    CSATerms,
    collateralize_netting_set,
    synthetic_netting_set_paths,
    uncollateralized_profile,
)


def main() -> None:
    times = np.arange(0, 8, dtype=float)
    paths = synthetic_netting_set_paths(times, n_paths=20_000, seed=37)

    uncollateralized = uncollateralized_profile(paths, times)
    csa = CSATerms(
        threshold_receive=500_000.0,
        threshold_post=500_000.0,
        minimum_transfer_amount=100_000.0,
        margin_period_steps=1,
    )
    collateralized = collateralize_netting_set(paths, times, csa)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    uncollateralized.to_csv(output_dir / "uncollateralized_profile.csv")
    collateralized.exposure_profile.to_csv(
        output_dir / "collateralized_profile.csv"
    )

    pd.DataFrame(
        [
            {
                "case": "uncollateralized",
                "mean_epe": uncollateralized["epe"].mean(),
                "terminal_pfe": uncollateralized["pfe"].iloc[-1],
                "terminal_mean_abs_collateral": 0.0,
            },
            {
                "case": "csa_collateralized",
                "mean_epe": collateralized.exposure_profile["epe"].mean(),
                "terminal_pfe": collateralized.exposure_profile["pfe"].iloc[-1],
                "terminal_mean_abs_collateral": (
                    collateralized.exposure_profile["mean_abs_collateral"].iloc[-1]
                ),
            },
        ]
    ).to_csv(output_dir / "collateral_summary.csv", index=False)

    print(collateralized.exposure_profile)


if __name__ == "__main__":
    main()
