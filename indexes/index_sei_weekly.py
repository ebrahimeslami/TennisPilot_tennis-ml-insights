# -*- coding: utf-8 -*-
"""
Seasonal Endurance Index (SEI)
------------------------------
Captures player resilience and sustained performance across a calendar year.

Components:
  - Match Density (MD)
  - Performance Retention (PR)
  - Late-Season Strength (LSS)
  - Recovery Factor (RF)

Outputs:
  - sei_match.csv
  - sei_weekly.csv
  - sei_weekly_surface.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------
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

# ---------------------------
# Build long player-match format
# ---------------------------
def build_long(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["iso_year"] = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"] = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time

    W = pd.DataFrame({
        "player_id": m["winner_id"].astype(str),
        "player_name": m["winner_name"],
        "opp_id": m["loser_id"].astype(str),
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "iso_year": m["iso_year"],
        "week_num": m["week_num"],
        "week_start": m["week_start"],
        "label": 1,
    })
    L = pd.DataFrame({
        "player_id": m["loser_id"].astype(str),
        "player_name": m["loser_name"],
        "opp_id": m["winner_id"].astype(str),
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "iso_year": m["iso_year"],
        "week_num": m["week_num"],
        "week_start": m["week_start"],
        "label": 0,
    })
    df = pd.concat([W, L], ignore_index=True)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df

# ---------------------------
# Core calculations
# ---------------------------
def compute_sei(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, grp in df.groupby(["player_id", "iso_year"], sort=False):
        g = grp.sort_values("date").copy()
        if len(g) < 5:
            continue

        # 1. Match density (scaled within year)
        matches_year = len(g)
        g["matches_year"] = matches_year

        # 2. Rolling 8-week win rate slope (Performance Retention)
        g["win_8w"] = g["label"].rolling(8, min_periods=3).mean()
        if g["win_8w"].notna().sum() > 5:
            x = np.arange(len(g["win_8w"].dropna()))
            y = g["win_8w"].dropna().values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
        else:
            slope = 0
        g["PR"] = 1 - abs(slope)

        # 3. Late-Season Strength (last 12 vs first 12 weeks)
        first = g.head(min(12, len(g)))
        last = g.tail(min(12, len(g)))
        LSS = last["label"].mean() - first["label"].mean()
        g["LSS"] = LSS

        # 4. Recovery Factor (wins after loss)
        losses = g["label"].shift(1) == 0
        wins_after_loss = (losses & (g["label"] == 1)).sum()
        RF = wins_after_loss / (losses.sum() + 1e-6)
        g["RF"] = RF

        # combine normalized components (weights)
        g["MD_norm"] = (matches_year - 20) / 60
        g["MD_norm"] = g["MD_norm"].clip(0, 1)
        g["PR_norm"] = (g["PR"] - g["PR"].min()) / (g["PR"].max() - g["PR"].min() + 1e-6)
        g["LSS_norm"] = (g["LSS"] - (-0.5)) / 1.0
        g["LSS_norm"] = g["LSS_norm"].clip(0, 1)
        g["RF_norm"] = g["RF"].clip(0, 1)

        g["SEI_match"] = (
            0.25 * g["MD_norm"] +
            0.30 * g["PR_norm"] +
            0.25 * g["LSS_norm"] +
            0.20 * g["RF_norm"]
        )

        out.append(g)

    if not out:
        return pd.DataFrame(columns=df.columns.tolist() + ["SEI_match"])
    return pd.concat(out, ignore_index=True)

# ---------------------------
# Weekly aggregations
# ---------------------------
def weekly_aggregate(df: pd.DataFrame, by_surface: bool=False) -> pd.DataFrame:
    cols = ["player_id", "player_name", "iso_year", "week_num", "week_start"]
    if by_surface:
        cols.append("surface")
    out = (
        df.groupby(cols, observed=True)
        .agg(matches=("label", "size"),
             SEI_mean=("SEI_match", "mean"),
             PR_mean=("PR_norm", "mean"),
             LSS_mean=("LSS_norm", "mean"),
             RF_mean=("RF_norm", "mean"))
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Seasonal Endurance Index (SEI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building long player-match table…")
    df = build_long(master)

    print("[INFO] Computing SEI per match…")
    df_sei = compute_sei(df)

    match_out = out_root / "sei_match.csv"
    df_sei.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level SEI → {match_out} ({len(df_sei):,} rows)")

    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(df_sei, by_surface=False)
    weekly.to_csv(out_root / "sei_weekly.csv", index=False)
    print(f"[INFO] Saved weekly SEI → {out_root/'sei_weekly.csv'} ({len(weekly):,} rows)")

    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(df_sei, by_surface=True)
    weekly_surf.to_csv(out_root / "sei_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface SEI → {out_root/'sei_weekly_surface.csv'} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
