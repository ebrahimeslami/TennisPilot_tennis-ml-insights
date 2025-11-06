# -*- coding: utf-8 -*-
"""
Rivalry Intensity Index (RII)
-----------------------------
Quantifies rivalry strength between player pairs (frequency, balance, competitiveness, recency, and tournament importance).

Outputs:
    - rii_match.csv
    - rii_weekly.csv
    - rii_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- helpers ----------
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---------- build long ----------
def build_long(master):
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["iso_year"] = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"] = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time
    return m

# ---------- compute RII ----------
def compute_rii(df):
    df = df.copy()
    df["pair_id"] = df.apply(lambda r: "_".join(sorted([str(r.winner_id), str(r.loser_id)])), axis=1)

    # frequency
    freq = df["pair_id"].value_counts().to_dict()
    df["freq"] = df["pair_id"].map(freq)
    df["freq_norm"] = df["freq"] / max(df["freq"].max(), 1)

    # balance (win ratio closeness)
    win_counts = df.groupby(["pair_id","winner_id"]).size().unstack(fill_value=0)
    wr = (win_counts.min(axis=1) / win_counts.sum(axis=1)).to_dict()
    df["balance"] = df["pair_id"].map(wr) * 2  # scaled 0-1

    # competitiveness proxy
    df["sets_played"] = df["score"].fillna("").apply(lambda s: len(s.split("-")) if isinstance(s,str) else np.nan)
    df["competitiveness"] = np.clip(df["sets_played"]/3, 0, 1)

    # recency weight
    latest = df.groupby("pair_id")["tourney_date"].transform("max")
    df["years_since"] = (latest - df["tourney_date"]).dt.days / 365.25
    df["recency"] = np.exp(-df["years_since"]/3)

    # tournament importance
    level_weight = {"G":1.0, "M":0.8, "A":0.6, "B":0.4, "D":0.3, "F":0.2}
    df["tourney_level_norm"] = df["tourney_level"].map(level_weight).fillna(0.3)

    # weighted sum
    wF,wC,wB,wR,wT = 0.25,0.25,0.25,0.15,0.10
    df["RII"] = (wF*df["freq_norm"] + wC*df["competitiveness"] +
                 wB*df["balance"] + wR*df["recency"] + wT*df["tourney_level_norm"]).clip(0,1)
    return df

# ---------- weekly aggregation ----------
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("RII","count"),
            avg_RII=("RII","mean"),
            avg_compet=("competitiveness","mean"),
            avg_balance=("balance","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute Rivalry Intensity Index (RII)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing RII...")
    df_rii = compute_rii(build_long(master))

    # match-level
    path_match = f"{args.out_root}/rii_match.csv"
    df_rii.to_csv(path_match, index=False)
    print(f"[INFO] Saved match-level RII → {path_match} ({len(df_rii):,} rows)")

    # weekly
    weekly = weekly_aggregate(df_rii, by_surface=False)
    wpath = f"{args.out_root}/rii_weekly.csv"
    weekly.to_csv(wpath, index=False)
    print(f"[INFO] Saved weekly RII → {wpath}")

    # surface
    weekly_surf = weekly_aggregate(df_rii, by_surface=True)
    wsurf = f"{args.out_root}/rii_weekly_surface.csv"
    weekly_surf.to_csv(wsurf, index=False)
    print(f"[INFO] Saved weekly surface RII → {wsurf}")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly.iso_year.min())}-{int(weekly.iso_year.max())} | Players: {weekly.winner_id.nunique()}")

if __name__ == "__main__":
    main()
