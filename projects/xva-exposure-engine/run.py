from dataclasses import replace
from pathlib import Path

import pandas as pd

from derivatives_xva.xva import default_problem, solve


def main() -> None:
    base_problem = default_problem()
    base = solve(base_problem)
    wwr = solve(replace(base_problem, wrong_way_beta=0.75))

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    base.exposure_profile.to_csv(output_dir / "exposure_profile.csv")

    pd.DataFrame(
        [
            {
                "case": "independent_credit",
                "fixed_rate": base.fixed_rate,
                "cva": base.cva,
                "dva": base.dva,
                "fva": base.fva,
                "mva": base.mva,
                "total_xva": base.total_xva,
            },
            {
                "case": "wrong_way_proxy",
                "fixed_rate": wwr.fixed_rate,
                "cva": wwr.cva,
                "dva": wwr.dva,
                "fva": wwr.fva,
                "mva": wwr.mva,
                "total_xva": wwr.total_xva,
            },
        ]
    ).to_csv(output_dir / "xva_summary.csv", index=False)

    print(base.to_dict())


if __name__ == "__main__":
    main()
