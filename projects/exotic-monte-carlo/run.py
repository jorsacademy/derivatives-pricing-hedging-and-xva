from pathlib import Path

import pandas as pd

from derivatives_xva.exotics import (
    arithmetic_asian_call_mc,
    european_call_mc,
    up_and_out_call_mc,
)
from derivatives_xva.options import black_scholes_price


def main() -> None:
    european = european_call_mc(
        100, 100, 1.0, 0.03, 0.20, n_paths=50_000, seed=29
    )
    asian = arithmetic_asian_call_mc(
        100, 100, 1.0, 0.03, 0.20, n_paths=50_000, seed=29
    )
    barrier = up_and_out_call_mc(
        100, 100, 125, 1.0, 0.03, 0.20, n_paths=50_000, seed=29
    )
    analytical = black_scholes_price(100, 100, 1.0, 0.03, 0.20, "call")

    table = pd.DataFrame(
        [
            {
                "instrument": "european_call_mc",
                "price": european.price,
                "standard_error": european.standard_error,
                "analytical_benchmark": analytical,
            },
            {
                "instrument": "arithmetic_asian_call",
                "price": asian.price,
                "standard_error": asian.standard_error,
                "analytical_benchmark": None,
            },
            {
                "instrument": "up_and_out_call",
                "price": barrier.price,
                "standard_error": barrier.standard_error,
                "analytical_benchmark": None,
            },
        ]
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    table.to_csv(output_dir / "exotic_price_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
