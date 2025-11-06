# -*- coding: utf-8 -*-
"""
Storyline Potential Index (SPI)
-------------------------------
Blends rivalry (PRII), competitiveness, pressure, form, stakes, and recency
to quantify narrative energy of each match.

Outputs:
    - spi_match.csv
    - spi_weekly.csv
    - spi_weekly_surface.csv
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

# ---- load contextual data ----
def load_context(path, value_col):
    df = pd.read_csv(path)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
        df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
    elif "year" in df.columns:
        df["iso_year"] = df["year"].astype(int)
        df["week_num"] = 1
    return df.rename(columns={value_col: "value"})

# ---- merge player context ----
def merge_context(master, df_ctx, label):
    merged = master.merge(
        df_ctx[["player_id","iso_year","week_num","value"]],
        left_on=["winner_id","iso_year","week_num"],
        right_on=["player_id","iso_year","week_num"],
        how="left"
    ).rename(columns={"value":f"{label}_winner"}).drop(columns=["player_id"], errors="ignore")
    merged = merged.merge(
        df_ctx[["player_id","iso_year","week_num","value"]],
        left_on=["loser_id","iso_year","week_num"],
        right_on=["player_id","iso_year","week_num"],
        how="left"
    ).rename(columns={"value":f"{label}_loser"}).drop(columns=["player_id"], errors="ignore")
    return merged

# ---- compute SPI ----
def compute_spi(df, prii_path):
    # merge PRII baseline
    df_prii = pd.read_csv(prii_path, usecols=["pair_id","PRII"]) if "pair_id" in pd.read_csv(prii_path, nrows=1).columns else None
    if df_prii is not None:
        df["pair_id"] = df.apply(lambda r: "_".join(sorted([str(r.winner_id), str(r.loser_id)])), axis=1)
        df = df.merge(df_prii, on="pair_id", how="left")
    else:
        df["PRII"] = 0.5

    # components
    df["sets_played"] = df["score"].fillna("").apply(lambda s: len(s.split("-")) if isinstance(s,str) else np.nan)
    df["competitiveness"] = np.clip(df["sets_played"]/3, 0, 1)
    df["pressure"] = df[["CPI_winner","CPI_loser"]].mean(axis=1)
    df["momentum_clash"] = 1 - abs(df["FTI_winner"] - df["FTI_loser"]).fillna(0)
    level_weight = {"G":1.0,"M":0.8,"A":0.6,"B":0.4,"D":0.3,"F":0.2}
    df["stakes"] = df["tourney_level"].map(level_weight).fillna(0.4)
    df["recency"] = np.exp(-(datetime.now() - df["tourney_date"]).dt.days/1095)

    # SPI formula
    wR,wC,wP,wM,wS,wT = 0.30,0.20,0.15,0.15,0.10,0.10
    df["SPI"] = (
        wR*df["PRII"].fillna(0.5) +
        wC*df["competitiveness"].fillna(0.5) +
        wP*df["pressure"].fillna(0.5) +
        wM*df["momentum_clash"].fillna(0.5) +
        wS*df["stakes"].fillna(0.5) +
        wT*df["recency"].fillna(0.5)
    ).clip(0,1)

    return df

# ---- weekly aggregation ----
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("SPI","count"),
            avg_SPI=("SPI","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Compute Storyline Potential Index (SPI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--fti", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--prii", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master["surface_c"] = master["surface"].map(canon_surface)
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    # merge contexts
    df_fti = load_context(args.fti, "FTI")
    df_cpi = load_context(args.cpi, "CPI_weighted")
    master = merge_context(master, df_fti, "FTI")
    master = merge_context(master, df_cpi, "CPI")

    print("[INFO] Computing SPI...")
    df_spi = compute_spi(master, args.prii)

    path_match = f"{args.out_root}/spi_match.csv"
    df_spi.to_csv(path_match, index=False)
    print(f"[INFO] Saved match-level SPI → {path_match}")

    weekly = weekly_aggregate(df_spi, by_surface=False)
    weekly_path = f"{args.out_root}/spi_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly SPI → {weekly_path}")

    weekly_surf = weekly_aggregate(df_spi, by_surface=True)
    weekly_surf_path = f"{args.out_root}/spi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface SPI → {weekly_surf_path}")

if __name__ == "__main__":
    main()
