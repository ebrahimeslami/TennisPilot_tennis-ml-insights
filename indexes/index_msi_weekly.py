# -*- coding: utf-8 -*-
"""
Media Storyline Index (MSI)
---------------------------
Extends SPI with popularity, rivalry legacy, prestige, and surprise.
Outputs:
    - msi_match.csv
    - msi_weekly.csv
    - msi_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---- Load SPI and PRII ----
def load_index(path, name):
    df = pd.read_csv(path)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
        df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
    if "year" in df.columns and "iso_year" not in df.columns:
        df["iso_year"] = df["year"]
        df["week_num"] = 1
    print(f"[INFO] Loaded {name} → {len(df)} rows")
    return df

# ---- Compute MSI ----
def compute_msi(df, spi_path, prii_path):
    spi = load_index(spi_path, "SPI")
    prii = load_index(prii_path, "PRII")

    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df["iso_year"] = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time
    df["surface_c"] = df["surface"].map(canon_surface)

    # merge storyline context
    df = df.merge(spi[["iso_year","week_num","winner_id","avg_SPI"]],
                  left_on=["winner_id","iso_year","week_num"],
                  right_on=["winner_id","iso_year","week_num"],
                  how="left").rename(columns={"avg_SPI":"SPI"})
    df = df.merge(prii[["winner_id","iso_year","week_num","avg_PRII"]],
                  on=["winner_id","iso_year","week_num"], how="left").rename(columns={"avg_PRII":"PRII"})

    # player popularity (rank-based)
    df["rank_winner"] = df["winner_rank"].fillna(200)
    df["rank_loser"]  = df["loser_rank"].fillna(200)
    df["pop_winner"]  = 1 - (df["rank_winner"] - 1)/200
    df["pop_loser"]   = 1 - (df["rank_loser"] - 1)/200
    df["PPI"] = (df["pop_winner"] + df["pop_loser"]) / 2

    # tournament prestige
    level_weight = {"G":1.0,"M":0.8,"A":0.6,"B":0.4,"D":0.3,"F":0.2}
    df["TP"] = df["tourney_level"].map(level_weight).fillna(0.4)

    # recency (6-month half-life)
    df["recency"] = np.exp(-(datetime.now() - df["tourney_date"]).dt.days / 180)

    # rivalry legacy
    df["RL"] = df["PRII"].fillna(0.5)

    # upset surprise
    df["US"] = abs(df["rank_loser"] - df["rank_winner"]) / 200

    # final MSI
    wSPI,wPPI,wTP,wRH,wRL,wUS = 0.30,0.25,0.15,0.10,0.10,0.10
    df["MSI"] = (
        wSPI*df["SPI"].fillna(0.5) +
        wPPI*df["PPI"].fillna(0.5) +
        wTP*df["TP"].fillna(0.5) +
        wRH*df["recency"].fillna(0.5) +
        wRL*df["RL"].fillna(0.5) +
        wUS*df["US"].fillna(0.5)
    ).clip(0,1)

    return df

# ---- Weekly aggregation ----
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("MSI","count"),
            avg_MSI=("MSI","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Compute Media Storyline Index (MSI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--spi", required=True)
    ap.add_argument("--prii", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing MSI...")
    df_msi = compute_msi(master, args.spi, args.prii)

    match_path = f"{args.out_root}/msi_match.csv"
    df_msi.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level MSI → {match_path}")

    weekly = weekly_aggregate(df_msi, by_surface=False)
    weekly_path = f"{args.out_root}/msi_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly MSI → {weekly_path}")

    weekly_surf = weekly_aggregate(df_msi, by_surface=True)
    weekly_surf_path = f"{args.out_root}/msi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface MSI → {weekly_surf_path}")

if __name__ == "__main__":
    main()


