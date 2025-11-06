# plot_cpi_trends.py
"""
Purpose:
--------
Visualize Clutch Pressure Index (CPI) results produced by index_cpi.py or
index_cpi_weekly.py.

You can plot either:
    • Yearly CPI (from cpi_weighted_surface.csv)
    • Weekly CPI (from cpi_weekly_weighted_surface.parquet)

Usage examples:
---------------
python plot_cpi_trends.py --data "D:\Tennis\data\indexes\cpi_weighted_surface.csv" --player "Novak Djokovic"
python plot_cpi_trends.py --data "D:\Tennis\data\indexes\cpi_weekly_weighted_surface.parquet" --player "Rafael Nadal"

What to expect:
---------------
• Produces a simple Matplotlib line plot of CPI over time for the chosen player.
• The red dashed line at 0.0 indicates neutral clutch performance.
• Points above zero = better under pressure; below zero = worse under pressure.
• Saves figure as PNG next to the data file.
"""
# plot_cpi_trends.py (fixed)
"""
Improved CPI plotting tool with automatic year/week parsing.
Fixes blank/flat plots caused by misread time columns.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path: Path) -> pd.DataFrame:
    """Load CPI data from CSV or Parquet."""
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Parquet.")
    return df


def plot_cpi(df: pd.DataFrame, player_name: str, out_dir: Path):
    """Plot CPI and CPI_weighted for a given player."""

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Detect which time column to use
    time_col = None
    for c in ["year", "week", "date"]:
        if c in df.columns:
            time_col = c
            break
    if not time_col:
        raise ValueError("No time column (year/week) found in dataset.")

    # Filter player
    player_df = df[df["player_name"].str.contains(player_name, case=False, na=False)]
    if player_df.empty:
        print(f"❌ No records found for player '{player_name}'")
        return

    # Convert time to numeric/datetime as needed
    if time_col == "year":
        # Ensure numeric years and proper order
        player_df[time_col] = pd.to_numeric(player_df[time_col], errors="coerce").astype("Int64")
        player_df = player_df.sort_values(time_col)
    else:
        player_df[time_col] = pd.to_datetime(player_df[time_col], errors="coerce")
        player_df = player_df.sort_values(time_col)

    # Drop invalid rows
    player_df = player_df.dropna(subset=[time_col, "cpi"])
    player_df = player_df.sort_values(time_col)

    # Plot setup
    plt.figure(figsize=(10, 5))
    plt.plot(player_df[time_col], player_df["cpi"], label="CPI (unweighted)", color="blue", linewidth=2)
    if "cpi_weighted" in player_df.columns:
        plt.plot(player_df[time_col], player_df["cpi_weighted"], label="CPI (weighted)", color="green", linewidth=2)

    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Clutch Pressure Index (CPI) Trend — {player_name}")
    plt.xlabel("Year / Week")
    plt.ylabel("CPI Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save and show
    out_path = out_dir / f"CPI_trend_{player_name.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ Saved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--player", required=True)
    args = ap.parse_args()

    data_path = Path(args.data)
    df = load_data(data_path)
    plot_cpi(df, args.player, data_path.parent)


if __name__ == "__main__":
    main()

