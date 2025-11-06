import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def compute_tcci_weekly(df, window_weeks=8):
    """Compute rolling weekly TCS consistency index."""
    records = []
    for pid, sub in tqdm(df.groupby("player_id"), desc="Computing weekly TCCI"):
        sub = sub.sort_values(["iso_year", "week"])
        sub["TCS_roll_std"] = sub["TCS"].rolling(window_weeks, min_periods=4).std()
        sub["TCS_roll_mean"] = sub["TCS"].rolling(window_weeks, min_periods=4).mean()
        sub["TCCI"] = 1 - (sub["TCS_roll_std"] / (sub["TCS_roll_mean"] + 1e-6))
        sub["TCCI"] = sub["TCCI"].clip(0, 1)
        records.append(sub)
    return pd.concat(records, ignore_index=True)

def compute_tcci_yearly(df):
    """Compute per-year specialization consistency."""
    yearly = (
        df.groupby(["player_id", "player_name", "iso_year"])
        .agg(TCS_mean=("TCS", "mean"), TCS_std=("TCS", "std"))
        .reset_index()
    )
    yearly["TCCI_yearly"] = 1 - (yearly["TCS_std"] / (yearly["TCS_mean"] + 1e-6))
    yearly["TCCI_yearly"] = yearly["TCCI_yearly"].clip(0, 1)
    return yearly

def compute_tcci_surface(df_surf):
    """Compute surface-based consistency."""
    surf = (
        df_surf.groupby(["player_id", "player_name", "iso_year", "surface"])
        .agg(TCS_surface_mean=("TCS_surface", "mean"),
             TCS_surface_std=("TCS_surface", "std"))
        .reset_index()
    )
    surf["TCCI_surface"] = 1 - (surf["TCS_surface_std"] / (surf["TCS_surface_mean"] + 1e-6))
    surf["TCCI_surface"] = surf["TCCI_surface"].clip(0, 1)
    return surf

def main():
    parser = argparse.ArgumentParser(description="Compute Tournament Category Consistency Index (TCCI)")
    parser.add_argument("--tcs_weekly", required=True, help="Path to TCS weekly file")
    parser.add_argument("--tcs_surface", required=True, help="Path to TCS weekly surface file")
    parser.add_argument("--out_root", required=True, help="Output folder")
    parser.add_argument("--window_weeks", type=int, default=8)
    args = parser.parse_args()

    print(f"[INFO] Loading TCS weekly → {args.tcs_weekly}")
    df_weekly = pd.read_csv(args.tcs_weekly)
    df_surf = pd.read_csv(args.tcs_surface)

    print("[INFO] Computing rolling weekly TCCI...")
    weekly_df = compute_tcci_weekly(df_weekly, window_weeks=args.window_weeks)
    weekly_out = Path(args.out_root) / "tcci_weekly.csv"
    weekly_df.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly TCCI → {weekly_out} ({len(weekly_df):,} rows)")

    print("[INFO] Computing yearly consistency...")
    yearly_df = compute_tcci_yearly(df_weekly)
    yearly_out = Path(args.out_root) / "tcci_yearly.csv"
    yearly_df.to_csv(yearly_out, index=False)
    print(f"[INFO] Saved yearly TCCI → {yearly_out} ({len(yearly_df):,} rows)")

    print("[INFO] Computing surface-based consistency...")
    surf_df = compute_tcci_surface(df_surf)
    surf_out = Path(args.out_root) / "tcci_surface.csv"
    surf_df.to_csv(surf_out, index=False)
    print(f"[INFO] Saved surface TCCI → {surf_out} ({len(surf_df):,} rows)")

    print(f"[INFO] Coverage: {df_weekly['iso_year'].min()}–{df_weekly['iso_year'].max()} | Players: {df_weekly['player_id'].nunique()}")

if __name__ == "__main__":
    main()
