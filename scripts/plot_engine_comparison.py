from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("results/raw/latest_engine_comparison_summary.csv")
FIGURE_DIR = Path("results/figures/engine_comparison")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    for setting_label in df["setting_label"].unique():
        subset = df[df["setting_label"] == setting_label]

        plt.figure(figsize=(10, 5))
        for engine in subset["engine_name"].unique():
            engine_df = subset[subset["engine_name"] == engine]
            plt.plot(
                engine_df["category"],
                engine_df["avg_latency_sec"],
                marker="o",
                label=engine,
            )
        plt.xlabel("Category")
        plt.ylabel("Average Latency (seconds)")
        plt.title(f"Latency Comparison by Category - {setting_label}")
        plt.xticks(rotation=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"latency_comparison_{setting_label}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        for engine in subset["engine_name"].unique():
            engine_df = subset[subset["engine_name"] == engine]
            plt.plot(
                engine_df["category"],
                engine_df["avg_tokens_per_sec"],
                marker="o",
                label=engine,
            )
        plt.xlabel("Category")
        plt.ylabel("Average Tokens/sec")
        plt.title(f"Throughput Comparison by Category - {setting_label}")
        plt.xticks(rotation=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"throughput_comparison_{setting_label}.png")
        plt.close()

    print(f"Saved comparison plots to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()