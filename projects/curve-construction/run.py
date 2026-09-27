from pathlib import Path

import numpy as np
import pandas as pd

from derivatives_xva.curves import bootstrap_par_swap_curve, par_swap_rate


def main() -> None:
    maturities = np.arange(1, 8, dtype=float)
    par_rates = np.array([0.031, 0.033, 0.035, 0.0365, 0.038, 0.039, 0.040])
    curve = bootstrap_par_swap_curve(maturities, par_rates)

    table = pd.DataFrame(
        {
            "maturity": maturities,
            "input_par_rate": par_rates,
            "zero_rate": curve.zero_rates,
            "discount_factor": curve.discount(maturities),
            "repriced_par_rate": [par_swap_rate(curve, int(t)) for t in maturities],
        }
    )
    table["repricing_error"] = table["repriced_par_rate"] - table["input_par_rate"]

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    table.to_csv(output_dir / "bootstrapped_curve.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
