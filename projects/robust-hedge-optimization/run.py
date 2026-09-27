from pathlib import Path

import pandas as pd

from derivatives_xva.hedge import default_problem, solve


def main() -> None:
    result = solve(default_problem())
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    result.hedge_notionals.rename("notional").to_csv(
        output_dir / "hedge_notionals.csv"
    )
    result.residual_stress_pnl.rename("residual_pnl").to_csv(
        output_dir / "residual_stress_pnl.csv"
    )
    pd.DataFrame(
        [
            {
                "unhedged_worst_loss": result.unhedged_worst_loss,
                "optimized_worst_loss": result.worst_loss,
                "worst_loss_reduction": result.worst_loss_reduction,
                "gross_notional": result.gross_notional,
                "hedge_cost": result.hedge_cost,
            }
        ]
    ).to_csv(output_dir / "hedge_summary.csv", index=False)

    print(result.to_dict())


if __name__ == "__main__":
    main()
