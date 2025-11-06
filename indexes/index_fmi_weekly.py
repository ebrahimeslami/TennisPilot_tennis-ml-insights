# -*- coding: utf-8 -*-
"""
Form Momentum Index (FMI)
-------------------------
Tracks player momentum over short-term windows using:
  - Match outcome (+1/-1)
  - Opponent quality (based on ranking)
  - Surface weighting
  - Exponential decay for recent matches
Outputs:
  fmi_match.csv, fmi_weekly.csv, fmi_weekly_surface.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------
def to_datetime_yyyymmdd(series):
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    t = s.strip().lower()
    if "hard" in t:  return "Hard"
    if "clay" in t:  return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t or "indoor" in t: return "Carpet"
    return "Other"

surface_weights = {"Hard":1.00, "Clay":0.95, "Grass":1.05, "Carpet":1.00, "Other":1.00}


# ---------------------------
# Build long-format dataset
# ---------------------------
def build_long(master):
    master = master.copy()
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master = master[master["tourney_date"].dt.year >= 1991].copy()
    master["surface_c"] = master["surface"].map(canon_surface)
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    W = pd.DataFrame({
        "player_id": master["winner_id"].astype(str),
        "player_name": master["winner_name"],
        "opp_id": master["loser_id"].astype(str),
        "opp_name": master["loser_name"],
        "opp_rank": master["loser_rank"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "label": 1,
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
    })
    L = pd.DataFrame({
        "player_id": master["loser_id"].astype(str),
        "player_name": master["loser_name"],
        "opp_id": master["winner_id"].astype(str),
        "opp_name": master["winner_name"],
        "opp_rank": master["winner_rank"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "label": -1,
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
    })

    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["player_id", "date"])
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)
    return df


# ---------------------------
# Compute match-level FMI
# ---------------------------
def compute_fmi(df, decay_lambda=0.3, window_matches=10):
    df = df.copy()
    N = 2000  # normalization cap for ranks

    def per_player(sub):
        sub = sub.sort_values("date").copy()
        sub["opp_rank"] = sub["opp_rank"].fillna(N)
        sub["Q"] = 1 - (sub["opp_rank"].clip(1, N) / N)  # opponent quality
        sub["surface_w"] = sub["surface"].map(surface_weights).fillna(1.0)
        sub["res_weighted"] = sub["label"] * sub["Q"] * sub["surface_w"]

        fmi_vals = []
        for i in range(len(sub)):
            recent = sub.iloc[max(0, i-window_matches):i]
            if recent.empty:
                fmi_vals.append(0.0)
                continue
            ages = np.arange(len(recent))[::-1]
            weights = np.exp(-decay_lambda * ages)
            fmi_vals.append(np.sum(recent["res_weighted"].values * weights) / np.sum(weights))
        sub["FMI_match"] = fmi_vals
        return sub

    return df.groupby("player_id", group_keys=False).apply(per_player).reset_index(drop=True)


# ---------------------------
# Weekly aggregations
# ---------------------------
def weekly_aggregate(df, by_surface=False):
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")
    g = df.groupby(cols, observed=True)
    out = (
        g.agg(
            matches=("label","size"),
            wins=("label", lambda x: (x > 0).sum()),
            losses=("label", lambda x: (x < 0).sum()),
            fmi_mean=("FMI_match","mean"),
            avg_opp_rank=("opp_rank","mean"),
            win_rate=("label", lambda x: np.mean(x > 0)),
        )
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    out["streak_score"] = out["wins"] - out["losses"]
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Form Momentum Index (FMI)")
    ap.add_argument("--master", required=True, help="Path to master parquet file")
    ap.add_argument("--out_root", required=True, help="Output directory for CSV files")
    ap.add_argument("--decay_lambda", type=float, default=0.3, help="Exponential decay factor")
    ap.add_argument("--window_matches", type=int, default=10, help="Number of matches for rolling window")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player–match data…")
    df = build_long(master)

    print("[INFO] Computing Form Momentum Index (FMI)…")
    df = compute_fmi(df, args.decay_lambda, args.window_matches)

    # Match-level output
    match_out = out_root / "fmi_match.csv"
    df.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level FMI → {match_out} ({len(df):,} rows)")

    # Weekly (global)
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(df, by_surface=False)
    weekly_out = out_root / "fmi_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly FMI → {weekly_out} ({len(weekly):,} rows)")

    # Weekly (surface)
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(df, by_surface=True)
    weekly_surf_out = out_root / "fmi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface FMI → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    print(f"[INFO] Coverage: {weekly['iso_year'].min()}–{weekly['iso_year'].max()} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
