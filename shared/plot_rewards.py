import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def find_monitor_files(log_dir: str) -> list[str]:
    pattern = os.path.join(log_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    # Monitor files start with a header line "#"
    return [f for f in files if os.path.isfile(f)]


def load_monitor_csv(path: str) -> pd.DataFrame:
    # SB3 Monitor CSV has a comment header line starting with '#'
    return pd.read_csv(path, comment="#")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs", help="Folder containing Monitor CSV logs")
    parser.add_argument("--out", type=str, default="assets/reward_vs_episode.png", help="Output image path")
    parser.add_argument("--window", type=int, default=20, help="Moving average window")
    args = parser.parse_args()

    files = find_monitor_files(args.log_dir)
    if not files:
        raise SystemExit(
            f"No CSV logs found under '{args.log_dir}'. "
            "Make sure you trained with monitor_dir='./logs/' and that logs exist."
        )

    # Load and concatenate all monitor csv files
    dfs = []
    for f in files:
        df = load_monitor_csv(f)
        # Expected columns: r (episode reward), l (episode length), t (time)
        if "r" in df.columns:
            dfs.append(df[["r"]].rename(columns={"r": "reward"}))
    if not dfs:
        raise SystemExit("Found CSV files but none contained reward column 'r'.")

    data = pd.concat(dfs, ignore_index=True)
    data["episode"] = range(1, len(data) + 1)
    data["reward_ma"] = data["reward"].rolling(args.window).mean()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure()
    plt.plot(data["episode"], data["reward"], label="Reward")
    plt.plot(data["episode"], data["reward_ma"], label=f"Moving Avg (window={args.window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved plot to: {args.out}")
    print(f"Episodes plotted: {len(data)}")


if __name__ == "__main__":
    main()
