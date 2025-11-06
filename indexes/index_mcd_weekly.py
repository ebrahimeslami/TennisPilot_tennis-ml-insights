# -*- coding: utf-8 -*-
"""
Mental Consistency Delta (MCD)
------------------------------
Measures stability in mental performance (CPI, MTI) across tournaments.

Components:
  - Tournament-Level Mental Stability (TMS)
  - Seasonal Mental Volatility (SMV)
  - Mental Recovery Factor (MRF)

Outputs:
  - mcd_match.csv
  - mcd_weekly.csv
  - mcd_weekly_surface.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------
# Helpers
# --------------------------------------------
def to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s: object) -> str:
    if not isinstance(s, str): return "Other"
    t = s.strip().lower()
    if "hard" in t:  return "Hard"
    if "clay" in t:  return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t or "indoor" in t: return "Carpet"
    return "Other"

# --------------------------------------------
# Load CPI + MTI merged data
# --------------------------------------------
def merge_mental_data(master, cpi_path, mti_path):
    df_cpi = pd.read_csv(cpi_path)
    df_mti = pd.read_csv(mti_path)

    # Normalize player_id type
    for d in (df_cpi, df_mti):
        d["player_id"] = d["player_id"].astype(str)

    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df["surface_c"] = df["surface"].map(canon_surface)
    df["iso_year"] = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    # --- Merge CPI (year, CPI_weighted) ---
    df = df.merge(
        df_cpi[["player_id", "year", "CPI_weighted"]],
        left_on=["winner_id", "iso_year"],
        right_on=["player_id", "year"],
        how="left"
    )
    df.rename(columns={"CPI_weighted": "CPI"}, inplace=True)

    # --- Merge MTI (iso_year, MTI) ---
    df = df.merge(
        df_mti[["player_id", "iso_year", "MTI"]],
        left_on=["winner_id", "iso_year"],
        right_on=["player_id", "iso_year"],
        how="left"
    )

    # Clean up
    df = df.dropna(subset=["CPI", "MTI"])
    df["player_id"] = df["winner_id"].astype(str)
    df["player_name"] = df["winner_name"]
    df = df.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)

    print(f"[INFO] CPI rows merged: {df['CPI'].notna().sum():,}")
    print(f"[INFO] MTI rows merged: {df['MTI'].notna().sum():,}")
    return df


# --------------------------------------------
# Core computation
# --------------------------------------------
def compute_mcd(df):
    all_out = []
    for pid, grp in df.groupby("player_id", sort=False):
        g = grp.sort_values("tourney_date").copy()
        g["CPI_diff"] = g["CPI"].diff()
        g["MTI_diff"] = g["MTI"].diff()

        # Tournament-Level Mental Stability
        g["TMS"] = 1 - (abs(g["CPI_diff"]) + abs(g["MTI_diff"])) / 2
        g["TMS"] = g["TMS"].clip(0, 1)

        # Seasonal Mental Volatility
        smv = 1 - ((np.nanstd(g["CPI"]) + np.nanstd(g["MTI"])) / 2)
        smv = np.clip(smv, 0, 1)
        g["SMV"] = smv

        # Mental Recovery Factor
        rebounds = ((g["CPI_diff"] > 0) | (g["MTI_diff"] > 0)).sum()
        total = (len(g) - 1)
        g["MRF"] = rebounds / (total + 1e-6)

        # Composite MCD
        g["MCD_match"] = 0.4 * g["TMS"] + 0.4 * g["SMV"] + 0.2 * g["MRF"]
        all_out.append(g)

    return pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()

# --------------------------------------------
# Weekly aggregation
# --------------------------------------------
def weekly_aggregate(df, by_surface=False):
    cols = ["player_id", "player_name", "iso_year", "week_num", "week_start"]
    if by_surface:
        cols.append("surface_c")

    return (
        df.groupby(cols, observed=True)
        .agg(matches=("TMS", "size"),
             MCD_mean=("MCD_match", "mean"),
             TMS_mean=("TMS", "mean"),
             SMV_mean=("SMV", "mean"),
             MRF_mean=("MRF", "mean"))
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )

# --------------------------------------------
# Main
# --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mental Consistency Delta (MCD)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--mti", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Merging CPI + MTI data…")
    df = merge_mental_data(master, args.cpi, args.mti)

    print("[INFO] Computing MCD per match…")
    df_mcd = compute_mcd(df)

    match_path = out_root / "mcd_match.csv"
    df_mcd.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level MCD → {match_path} ({len(df_mcd):,} rows)")

    print("[INFO] Aggregating weekly global…")
    weekly = weekly_aggregate(df_mcd, by_surface=False)
    weekly.to_csv(out_root / "mcd_weekly.csv", index=False)
    print(f"[INFO] Saved weekly MCD → {out_root/'mcd_weekly.csv'} ({len(weekly):,} rows)")

    print("[INFO] Aggregating weekly by surface…")
    weekly_surf = weekly_aggregate(df_mcd, by_surface=True)
    weekly_surf.to_csv(out_root / "mcd_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface MCD → {out_root/'mcd_weekly_surface.csv'} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()

