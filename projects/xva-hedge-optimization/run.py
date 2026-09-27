from pathlib import Path

import pandas as pd

from derivatives_xva.xva_hedge import (
    benchmark_max_normalized_residual,
    default_problem,
    solve,
)


def main() -> None:
    problem = default_problem()
    result = solve(problem)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    problem.base_sensitivities.rename("sensitivity").to_csv(
        output_dir / "measured_xva_sensitivities.csv"
    )
    result.hedge_notionals.rename("notional").to_csv(
        output_dir / "xva_hedge_notionals.csv"
    )
    result.residual_sensitivities.rename("residual_sensitivity").to_csv(
        output_dir / "residual_xva_sensitivities.csv"
    )

    pd.DataFrame(
        [
            {
                "unhedged_max_normalized_residual": (
                    benchmark_max_normalized_residual(problem)
                ),
                "optimized_max_normalized_residual": (
                    result.max_normalized_residual
                ),
                "gross_notional": result.gross_notional,
                "hedge_cost": result.hedge_cost,
                "objective_value": result.objective_value,
            }
        ]
    ).to_csv(output_dir / "xva_hedge_summary.csv", index=False)

    print(result.hedge_notionals)
    print(result.residual_sensitivities)


if __name__ == "__main__":
    main()
