import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    input_file = sys.argv[1] if len(sys.argv) > 1 else "heston_smile.csv"

    strikes = []
    iv_mc = []
    iv_analytical = []

    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strikes.append(float(row["strike"]))
            iv_mc.append(float(row["iv_mc"]))
            iv_analytical.append(float(row["iv_analytical"]))

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, iv_mc, marker="o", linewidth=1.8, label="MC implied vol")
    plt.plot(
        strikes,
        iv_analytical,
        marker="s",
        linewidth=1.8,
        label="Analytical implied vol",
    )
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.title("Heston volatility smile")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("heston_smile.png", dpi=160)
    print("Saved heston_smile.png")


if __name__ == "__main__":
    main()
