# -*- coding: utf-8 -*-
"""
Return-Efficiency Index (REI)
-----------------------------
Quantifies return performance with four components:
  R1: First-serve return efficiency
  R2: Second-serve return efficiency
  R3: Return stability (volatility penalty; 52w std/mean)
  R4: Break-point conversion (breaks made / chances)

All components are bounded to [0,1] as appropriate and normalized
against prior 52-week (365D) baselines (shifted to avoid lookahead).
Outputs:
  - rei_match.csv
  - rei_weekly.csv
  - rei_weekly_surface.csv
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
# Build long player-match view
# ---------------------------
def build_long(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m.dropna(subset=["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"]  = m["surface"].map(canon_surface)
    m["iso_year"]   = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"]   = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Winner perspective row (player's return = opponent's serve stats)
    W = pd.DataFrame({
        "player_id":   m["winner_id"].astype(str),
        "player_name": m["winner_name"],
        "opp_id":      m["loser_id"].astype(str),
        "opp_name":    m["loser_name"],
        "date":        m["tourney_date"],
        "surface":     m["surface_c"],
        "iso_year":    m["iso_year"],
        "week_num":    m["week_num"],
        "week_start":  m["week_start"],
        "opp_svpt":    m["l_svpt"],
        "opp_1stIn":   m["l_1stIn"],
        "opp_1stWon":  m["l_1stWon"],
        "opp_2ndWon":  m["l_2ndWon"],
        "opp_bpFaced": m["l_bpFaced"],  # opponent faced BPs while serving
        "opp_bpSaved": m["l_bpSaved"],
        "label":       1  # won match
    })

    # Loser perspective row
    L = pd.DataFrame({
        "player_id":   m["loser_id"].astype(str),
        "player_name": m["loser_name"],
        "opp_id":      m["winner_id"].astype(str),
        "opp_name":    m["winner_name"],
        "date":        m["tourney_date"],
        "surface":     m["surface_c"],
        "iso_year":    m["iso_year"],
        "week_num":    m["week_num"],
        "week_start":  m["week_start"],
        "opp_svpt":    m["w_svpt"],
        "opp_1stIn":   m["w_1stIn"],
        "opp_1stWon":  m["w_1stWon"],
        "opp_2ndWon":  m["w_2ndWon"],
        "opp_bpFaced": m["w_bpFaced"],
        "opp_bpSaved": m["w_bpSaved"],
        "label":       0  # lost match
    })

    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["player_id","date"])
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)
    return df


# ---------------------------
# Compute raw return components
# ---------------------------
def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Return points won on 1st serve: opp_1stIn - opp_1stWon
    d["R1_raw"] = np.where(d["opp_1stIn"] > 0,
                           (d["opp_1stIn"] - d["opp_1stWon"]) / d["opp_1stIn"], np.nan)

    # Return points won on 2nd serve: (opp_svpt - opp_1stIn) - opp_2ndWon
    d["R2_raw"] = np.where((d["opp_svpt"] - d["opp_1stIn"]) > 0,
                           ((d["opp_svpt"] - d["opp_1stIn"]) - d["opp_2ndWon"]) / (d["opp_svpt"] - d["opp_1stIn"]), np.nan)

    # Break conversion: breaks made / chances = (opp_bpFaced - opp_bpSaved) / opp_bpFaced
    d["breaks_made"] = (d["opp_bpFaced"] - d["opp_bpSaved"]).clip(lower=0)
    d["R4_raw"] = np.where(d["opp_bpFaced"] > 0,
                           d["breaks_made"] / d["opp_bpFaced"], np.nan)

    # Total return points won rate (for stability)
    d["rpw_raw"] = np.where(d["opp_svpt"] > 0,
                            ((d["opp_svpt"] - d["opp_1stWon"] - d["opp_2ndWon"]) / d["opp_svpt"]), np.nan)

    # Clip to [0,1]
    for c in ["R1_raw","R2_raw","R4_raw","rpw_raw"]:
        d[c] = d[c].clip(0, 1)

    return d


# ---------------------------
# Add 52-week baselines and return stability
# ---------------------------
def add_rolling_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, sub in df.groupby("player_id", sort=False):
        s = sub.sort_values("date").set_index("date")

        # shift one match to use prior-only info
        for col in ["R1_raw","R2_raw","R4_raw","rpw_raw"]:
            s[col + "_prior"] = s[col].shift(1)

        # time-based rolling windows (365D)
        s["R1_base"]  = s["R1_raw_prior"].rolling("365D", min_periods=8).mean()
        s["R2_base"]  = s["R2_raw_prior"].rolling("365D", min_periods=8).mean()
        s["R4_base"]  = s["R4_raw_prior"].rolling("365D", min_periods=8).mean()

        s["rpw_mean"] = s["rpw_raw_prior"].rolling("365D", min_periods=8).mean()
        s["rpw_std"]  = s["rpw_raw_prior"].rolling("365D", min_periods=8).std()

        # Stability term: 1 - std/mean
        s["R3_raw"] = 1 - (s["rpw_std"] / (s["rpw_mean"] + 1e-6))
        s["R3_raw"] = s["R3_raw"].clip(0, 1)

        # Conservative defaults if no baseline yet
        s["R1_base"] = s["R1_base"].fillna(0.30)  # typical 1st-serve return rate ~30%
        s["R2_base"] = s["R2_base"].fillna(0.50)  # typical 2nd-serve return rate ~50%
        s["R4_base"] = s["R4_base"].fillna(0.35)  # typical BP conversion around mid-30s

        out.append(s.reset_index())

    return pd.concat(out, ignore_index=True)


# ---------------------------
# Compute REI per match
# ---------------------------
def compute_rei(df: pd.DataFrame, w1=0.30, w2=0.30, w3=0.20, w4=0.20) -> pd.DataFrame:
    d = df.copy()

    # Normalized (component - baseline) for R1, R2, R4
    d["R1_norm"] = d["R1_raw"] - d["R1_base"]
    d["R2_norm"] = d["R2_raw"] - d["R2_base"]
    d["R4_norm"] = d["R4_raw"] - d["R4_base"]

    weights = np.vstack([
        (~d["R1_norm"].isna()).astype(float) * w1,
        (~d["R2_norm"].isna()).astype(float) * w2,
        (~d["R3_raw"].isna()).astype(float)  * w3,
        (~d["R4_norm"].isna()).astype(float) * w4,
    ]).T
    wsum = weights.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)

    comps = np.vstack([
        d["R1_norm"].fillna(0.0),
        d["R2_norm"].fillna(0.0),
        d["R3_raw"].fillna(0.0),
        d["R4_norm"].fillna(0.0),
    ]).T

    d["REI_match"] = (weights * comps).sum(axis=1) / wsum
    return d


# ---------------------------
# Weekly aggregations
# ---------------------------
def weekly_aggregate(df: pd.DataFrame, by_surface: bool=False) -> pd.DataFrame:
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")
    g = df.groupby(cols, observed=True)
    out = (
        g.agg(
            matches=("label","size"),
            REI_mean=("REI_match","mean"),
            R1_mean=("R1_norm","mean"),
            R2_mean=("R2_norm","mean"),
            R3_mean=("R3_raw","mean"),
            R4_mean=("R4_norm","mean"),
        )
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Return-Efficiency Index (REI)")
    ap.add_argument("--master", required=True, help="Path to master parquet (e.g., D:\\Tennis\\data\\master\\tennis_master_1991.parquet)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--w1", type=float, default=0.30)
    ap.add_argument("--w2", type=float, default=0.30)
    ap.add_argument("--w3", type=float, default=0.20)
    ap.add_argument("--w4", type=float, default=0.20)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player–match table…")
    df = build_long(master)

    print("[INFO] Computing return components…")
    df = compute_components(df)

    print("[INFO] Adding 52-week baselines & stability…")
    df = add_rolling_baselines(df)

    print("[INFO] Computing REI per match…")
    df = compute_rei(df, args.w1, args.w2, args.w3, args.w4)

    # Match-level export
    match_cols = [
        "player_id","player_name","opp_id","opp_name",
        "date","surface","iso_year","week_num","week_start",
        "opp_svpt","opp_1stIn","opp_1stWon","opp_2ndWon","opp_bpFaced","opp_bpSaved",
        "R1_raw","R2_raw","R3_raw","R4_raw",
        "R1_base","R2_base","R4_base",
        "R1_norm","R2_norm","R4_norm",
        "REI_match"
    ]
    match_out = out_root / "rei_match.csv"
    df[match_cols].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level REI → {match_out} ({len(df):,} rows)")

    # Weekly global
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(df, by_surface=False)
    weekly_out = out_root / "rei_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly REI → {weekly_out} ({len(weekly):,} rows)")

    # Weekly by surface
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(df, by_surface=True)
    weekly_surf_out = out_root / "rei_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface REI → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
