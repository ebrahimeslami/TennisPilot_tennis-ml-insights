"""
Combine yearly ATP/WTA tennis match CSVs into one master parquet file.

Usage example (from D:\Tennis\data\Index):

python build_master.py --raw_root "D:\Tennis\data\TML-Database-master" --out_root "D:\Tennis\data" --start_year 1991
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_year_files(raw_root: Path, start_year: int):
    """Load all yearly CSVs starting from the given year."""
    files = sorted(raw_root.glob("**/*.csv"))
    dfs = []
    for f in files:
        try:
            year = int(f.stem[-4:])
        except Exception:
            year = None
        if year is not None and year >= start_year:
            print(f"Loading {f.name} ...")
            df = pd.read_csv(f)
            df["year"] = year
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No CSV files found from {start_year} onwards in {raw_root}")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all):,} rows from {len(dfs)} yearly files.")
    return df_all

def normalize_columns(df: pd.DataFrame):
    """Normalize core columns so all years have consistent schema."""
    rename_map = {
        "tourney_date": "tourney_date",
        "tourney_name": "tourney_name",
        "tourney_level": "tourney_level",
        "surface": "surface",
        "draw_size": "draw_size",
        "best_of": "best_of",
        "minutes": "minutes",
        "winner_id": "winner_id",
        "winner_name": "winner_name",
        "winner_hand": "winner_hand",
        "winner_ht": "winner_ht",
        "winner_age": "winner_age",
        "winner_rank": "winner_rank",
        "winner_rank_points": "winner_rank_points",
        "loser_id": "loser_id",
        "loser_name": "loser_name",
        "loser_hand": "loser_hand",
        "loser_ht": "loser_ht",
        "loser_age": "loser_age",
        "loser_rank": "loser_rank",
        "loser_rank_points": "loser_rank_points",
        "w_ace": "w_ace",
        "w_df": "w_df",
        "w_svpt": "w_svpt",
        "w_1stIn": "w_1stIn",
        "w_1stWon": "w_1stWon",
        "w_2ndWon": "w_2ndWon",
        "w_SvGms": "w_SvGms",
        "w_bpSaved": "w_bpSaved",
        "w_bpFaced": "w_bpFaced",
        "l_ace": "l_ace",
        "l_df": "l_df",
        "l_svpt": "l_svpt",
        "l_1stIn": "l_1stIn",
        "l_1stWon": "l_1stWon",
        "l_2ndWon": "l_2ndWon",
        "l_SvGms": "l_SvGms",
        "l_bpSaved": "l_bpSaved",
        "l_bpFaced": "l_bpFaced",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def add_efficiency_metrics(df: pd.DataFrame):
    """Add derived performance metrics."""
    # Winner serve efficiency
    df["w_first_pct"] = np.where(df["w_svpt"] > 0, df["w_1stIn"] / df["w_svpt"], np.nan)
    df["w_firstwon_pct"] = np.where(df["w_1stIn"] > 0, df["w_1stWon"] / df["w_1stIn"], np.nan)
    df["w_secondwon_pct"] = np.where(df["w_svpt"] > df["w_1stIn"], df["w_2ndWon"] / (df["w_svpt"] - df["w_1stIn"]), np.nan)
    df["w_bp_save_ratio"] = np.where(df["w_bpFaced"] > 0, df["w_bpSaved"] / df["w_bpFaced"], np.nan)
    df["w_se"] = (df["w_firstwon_pct"] * 0.7 + df["w_secondwon_pct"] * 0.3)

    # Loser serve efficiency
    df["l_first_pct"] = np.where(df["l_svpt"] > 0, df["l_1stIn"] / df["l_svpt"], np.nan)
    df["l_firstwon_pct"] = np.where(df["l_1stIn"] > 0, df["l_1stWon"] / df["l_1stIn"], np.nan)
    df["l_secondwon_pct"] = np.where(df["l_svpt"] > df["l_1stIn"], df["l_2ndWon"] / (df["l_svpt"] - df["l_1stIn"]), np.nan)
    df["l_bp_save_ratio"] = np.where(df["l_bpFaced"] > 0, df["l_bpSaved"] / df["l_bpFaced"], np.nan)
    df["l_se"] = (df["l_firstwon_pct"] * 0.7 + df["l_secondwon_pct"] * 0.3)

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="Path to root folder containing yearly CSVs")
    ap.add_argument("--out_root", type=str, required=True, help="Output folder (e.g., D:\\Tennis\\data)")
    ap.add_argument("--start_year", type=int, default=1991)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_dir = out_root / "master"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_year_files(raw_root, args.start_year)
    df = normalize_columns(df)
    df = add_efficiency_metrics(df)

    # Determine gender from file path
    df["gender"] = np.where(df["tourney_id"].astype(str).str.startswith("W"), "WTA", "ATP")

    out_file = out_dir / f"tennis_master_{args.start_year}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"\n✅ Master parquet saved to: {out_file}")
    print(f"Rows: {len(df):,}, Years: {df['year'].min()}–{df['year'].max()}")

if __name__ == "__main__":
    main()
