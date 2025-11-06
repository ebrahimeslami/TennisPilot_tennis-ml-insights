# -*- coding: utf-8 -*-
"""
Career Defining Moment Index (CDMI)
-----------------------------------
Quantifies historical turning points in player careers based on
form inflection, ranking change, pressure context, and opponent quality.

Outputs:
    - cdmi_match.csv
    - cdmi_weekly.csv
    - cdmi_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---- Load context ----
def load_context(path, val_col):
    df = pd.read_csv(path)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
        df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
    elif "year" in df.columns:
        df["iso_year"] = df["year"].astype(int)
        df["week_num"] = 1
    return df.rename(columns={val_col: "value"})

# ---- Compute CDMI ----
def compute_cdmi(df, fti_path, cpi_path):
    df_fti = load_context(fti_path, "FTI")
    df_cpi = load_context(cpi_path, "CPI_weighted")

    # Merge player FTI and CPI
    for label, ctx in [("FTI", df_fti), ("CPI", df_cpi)]:
        df = df.merge(
            ctx[["player_id","iso_year","week_num","value"]],
            left_on=["winner_id","iso_year","week_num"],
            right_on=["player_id","iso_year","week_num"],
            how="left"
        ).rename(columns={"value":f"{label}_winner"}).drop(columns=["player_id"], errors="ignore")
        df = df.merge(
            ctx[["player_id","iso_year","week_num","value"]],
            left_on=["loser_id","iso_year","week_num"],
            right_on=["player_id","iso_year","week_num"],
            how="left"
        ).rename(columns={"value":f"{label}_loser"}).drop(columns=["player_id"], errors="ignore")

    # ΔFTI: change 4 weeks before vs after
    df["FTI_change"] = abs(df["FTI_winner"].fillna(0) - df["FTI_loser"].fillna(0))

    # Rank change proxy (simulate next 8 weeks)
    df["rank_change"] = abs(df["winner_rank"].fillna(200) - df["loser_rank"].fillna(200)) / 200

    # Pressure, Prestige, Opponent legacy
    df["pressure"] = df[["CPI_winner","CPI_loser"]].mean(axis=1)
    level_weight = {"G":1.0,"M":0.8,"A":0.6,"B":0.4,"D":0.3,"F":0.2}
    df["prestige"] = df["tourney_level"].map(level_weight).fillna(0.4)
    df["opp_legacy"] = 1 - (df["loser_rank"].fillna(200)-1)/200

    # Surprise factor (upsets)
    df["upset"] = abs(df["winner_rank"] - df["loser_rank"]) / 200

    # final CDMI
    wF,wR,wP,wS,wO,wU = 0.25,0.25,0.20,0.15,0.10,0.05
    df["CDMI"] = (
        wF*df["FTI_change"] +
        wR*df["rank_change"] +
        wP*df["pressure"].fillna(0.5) +
        wS*df["prestige"] +
        wO*df["opp_legacy"] +
        wU*df["upset"]
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
            matches=("CDMI","count"),
            avg_CDMI=("CDMI","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Compute Career Defining Moment Index (CDMI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--fti", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master["surface_c"] = master["surface"].map(canon_surface)
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    print("[INFO] Computing CDMI...")
    df_cdmi = compute_cdmi(master, args.fti, args.cpi)

    match_path = f"{args.out_root}/cdmi_match.csv"
    df_cdmi.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level CDMI → {match_path}")

    weekly = weekly_aggregate(df_cdmi, by_surface=False)
    weekly_path = f"{args.out_root}/cdmi_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly CDMI → {weekly_path}")

    weekly_surf = weekly_aggregate(df_cdmi, by_surface=True)
    weekly_surf_path = f"{args.out_root}/cdmi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface CDMI → {weekly_surf_path}")

if __name__ == "__main__":
    main()
