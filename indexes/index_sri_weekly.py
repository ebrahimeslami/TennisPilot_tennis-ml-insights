# -*- coding: utf-8 -*-
"""
Serve-Reliability Index (SRI)
-----------------------------
Measures consistency and efficiency of a player's serve using:
  1. First Serve Effectiveness (S1)
  2. Second Serve Resilience (S2)
  3. Serve Stability (S3)
  4. Break Protection (S4)

Each component is normalized relative to the player's 52-week baseline,
then combined into a weighted composite (SRI_match). Outputs include:
  - sri_match.csv
  - sri_weekly.csv
  - sri_weekly_surface.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def to_datetime_yyyymmdd(series):
    """Convert integer YYYYMMDD values to datetime."""
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str):
        return "Other"
    t = s.strip().lower()
    if "hard" in t:  return "Hard"
    if "clay" in t:  return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t or "indoor" in t: return "Carpet"
    return "Other"

# ----------------------------------------------------------------------
# Build long-format player–match dataset
# ----------------------------------------------------------------------
def build_long(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master = master.dropna(subset=["tourney_date"])
    master = master[master["tourney_date"].dt.year >= 1991].copy()

    master["surface_c"] = master["surface"].map(canon_surface)
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Winner records
    W = pd.DataFrame({
        "player_id": master["winner_id"].astype(str),
        "player_name": master["winner_name"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "svpt": master["w_svpt"],
        "first_in": master["w_1stIn"],
        "first_won": master["w_1stWon"],
        "second_won": master["w_2ndWon"],
        "bp_saved": master["w_bpSaved"],
        "bp_faced": master["w_bpFaced"],
        "label": 1
    })

    # Loser records
    L = pd.DataFrame({
        "player_id": master["loser_id"].astype(str),
        "player_name": master["loser_name"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "svpt": master["l_svpt"],
        "first_in": master["l_1stIn"],
        "first_won": master["l_1stWon"],
        "second_won": master["l_2ndWon"],
        "bp_saved": master["l_bpSaved"],
        "bp_faced": master["l_bpFaced"],
        "label": 0
    })

    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["player_id", "date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df

# ----------------------------------------------------------------------
# Compute serve components
# ----------------------------------------------------------------------
def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw serve performance components per match."""
    df = df.copy()

    # Avoid division by zero and unrealistic ratios
    df["S1_raw"] = np.where(df["first_in"] > 0,
                            df["first_won"] / df["first_in"], np.nan)
    df["S2_raw"] = np.where((df["svpt"] - df["first_in"]) > 0,
                            df["second_won"] / (df["svpt"] - df["first_in"]), np.nan)
    df["S4_raw"] = np.where(df["bp_faced"] > 0,
                            df["bp_saved"] / df["bp_faced"], np.nan)

    # Clip to [0,1]
    for c in ["S1_raw", "S2_raw", "S4_raw"]:
        df[c] = df[c].clip(0, 1)

    return df

# ----------------------------------------------------------------------
# Add 52-week baselines and stability term
# ----------------------------------------------------------------------
def add_rolling_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, sub in df.groupby("player_id", sort=False):
        s = sub.sort_values("date").set_index("date")

        # Shifted rolling mean and std over 365 days
        for col in ["S1_raw", "S2_raw", "S4_raw"]:
            s[col + "_prior"] = s[col].shift(1)
            s[col + "_base"] = s[col + "_prior"].rolling("365D", min_periods=8).mean()
            s[col + "_std"] = s[col + "_prior"].rolling("365D", min_periods=8).std()

        # Serve stability = 1 - (rolling std / rolling mean)
        s["S3_raw"] = 1 - (s["S1_raw_std"] / (s["S1_raw_base"] + 1e-6))
        s["S3_raw"] = s["S3_raw"].clip(0, 1)

        # fill missing baselines with conservative means
        for col in ["S1_raw_base", "S2_raw_base", "S4_raw_base"]:
            s[col] = s[col].fillna(0.5)

        out.append(s.reset_index())

    return pd.concat(out, ignore_index=True)

# ----------------------------------------------------------------------
# Compute SRI per match
# ----------------------------------------------------------------------
def compute_sri(df: pd.DataFrame,
                w1=0.30, w2=0.25, w3=0.25, w4=0.20) -> pd.DataFrame:
    df = df.copy()

    # Normalized deltas
    df["S1_norm"] = df["S1_raw"] - df["S1_raw_base"]
    df["S2_norm"] = df["S2_raw"] - df["S2_raw_base"]
    df["S4_norm"] = df["S4_raw"] - df["S4_raw_base"]

    # Combine
    weights = np.vstack([
        (~df["S1_norm"].isna()).astype(float) * w1,
        (~df["S2_norm"].isna()).astype(float) * w2,
        (~df["S3_raw"].isna()).astype(float) * w3,
        (~df["S4_norm"].isna()).astype(float) * w4,
    ]).T

    wsum = weights.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)

    comps = np.vstack([
        df["S1_norm"].fillna(0.0),
        df["S2_norm"].fillna(0.0),
        df["S3_raw"].fillna(0.0),
        df["S4_norm"].fillna(0.0),
    ]).T

    df["SRI_match"] = (weights * comps).sum(axis=1) / wsum
    return df

# ----------------------------------------------------------------------
# Weekly aggregations
# ----------------------------------------------------------------------
def weekly_aggregate(df: pd.DataFrame, by_surface=False) -> pd.DataFrame:
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")
    g = df.groupby(cols, observed=True)
    out = (
        g.agg(
            matches=("label","size"),
            SRI_mean=("SRI_match","mean"),
            S1_mean=("S1_norm","mean"),
            S2_mean=("S2_norm","mean"),
            S3_mean=("S3_raw","mean"),
            S4_mean=("S4_norm","mean"),
        )
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    return out

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Serve-Reliability Index (SRI)")
    ap.add_argument("--master", required=True, help="Path to master parquet")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--w1", type=float, default=0.30)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--w3", type=float, default=0.25)
    ap.add_argument("--w4", type=float, default=0.20)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player–match table…")
    df = build_long(master)

    print("[INFO] Computing raw serve components…")
    df = compute_components(df)

    print("[INFO] Adding 52-week rolling baselines…")
    df = add_rolling_baselines(df)

    print("[INFO] Computing SRI per match…")
    df = compute_sri(df, args.w1, args.w2, args.w3, args.w4)

    # --- Match-level output ---
    match_cols = [
        "player_id","player_name","date","surface","iso_year","week_num","week_start",
        "svpt","first_in","first_won","second_won","bp_saved","bp_faced",
        "S1_raw","S2_raw","S3_raw","S4_raw",
        "S1_raw_base","S2_raw_base","S4_raw_base",
        "S1_norm","S2_norm","S4_norm","SRI_match"
    ]
    match_out = out_root / "sri_match.csv"
    df[match_cols].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level SRI → {match_out} ({len(df):,} rows)")

    # --- Weekly global ---
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(df, by_surface=False)
    weekly_out = out_root / "sri_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly SRI → {weekly_out} ({len(weekly):,} rows)")

    # --- Weekly by surface ---
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(df, by_surface=True)
    weekly_surf_out = out_root / "sri_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface SRI → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    print(f"[INFO] Coverage: {weekly['iso_year'].min()}–{weekly['iso_year'].max()} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
