from pathlib import Path

import pandas as pd

from derivatives_xva.options import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)


def main() -> None:
    rows = []
    for strike in [80, 90, 100, 110, 120]:
        price = black_scholes_price(100, strike, 1.0, 0.03, 0.24, "call")
        greeks = black_scholes_greeks(100, strike, 1.0, 0.03, 0.24, "call")
        iv = implied_volatility(price, 100, strike, 1.0, 0.03, "call")
        rows.append(
            {
                "strike": strike,
                "price": price,
                "implied_volatility": iv,
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "vega": greeks.vega,
                "theta": greeks.theta,
            }
        )

    table = pd.DataFrame(rows)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    table.to_csv(output_dir / "option_greeks.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
