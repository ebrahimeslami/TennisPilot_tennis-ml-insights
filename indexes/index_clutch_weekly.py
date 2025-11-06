"""
Clutch Performance Index (CPI-Advanced)
========================================

Purpose:
--------
Quantifies how well a player performs in high-pressure moments
(break points, tiebreaks, deciding sets), adjusted for tournament level.

Metrics:
---------
For each match:
- BP_save_rate     = w_bpSaved / w_bpFaced
- BP_convert_rate  = (opponent break points faced - opponent break points saved) / opponent break points faced
- TB_rate          = proxy using set scores or tiebreak count if available

Composite Index:
----------------
CPI_adv = 0.4*(BP_save_rate - 0.5)
        + 0.4*(BP_convert_rate - 0.5)
        + 0.2*(TB_rate - 0.5)
CPI_weighted = CPI_adv * tier_weight

Weekly aggregation:
-------------------
Mean of CPI_weighted per player-week.

Outputs:
---------
- clutch_match.csv
- clutch_weekly.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------
# Tournament tier weights
# -------------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}

def _normalize_tourney_level(x: str) -> str:
    if not isinstance(x, str):
        return "B"
    t = x.strip().upper()
    if "G" in t or "SLAM" in t:
        return "G"
    if "M" in t or "1000" in t or "MAST" in t:
        return "M"
    if "A" in t or "500" in t:
        return "A"
    return "B"

def _to_week(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("W-MON").dt.start_time

# --------------------------------
# Build player-match dataframe
# --------------------------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # convert tourney_date to datetime
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # normalize tournament level and assign weights
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)
    df["tier_w"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    df["week"] = _to_week(df["tourney_date"])
    df["surface"] = df["surface"].fillna("Hard").astype(str)

    # Build winner-side stats
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "label": 1,
        "date": df["tourney_date"],
        "week": df["week"],
        "surface": df["surface"],
        "tier_w": df["tier_w"],
        "bp_saved": pd.to_numeric(df["w_bpSaved"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["w_bpFaced"], errors="coerce"),
        "opp_bp_saved": pd.to_numeric(df["l_bpSaved"], errors="coerce"),
        "opp_bp_faced": pd.to_numeric(df["l_bpFaced"], errors="coerce"),
        "score": df["score"]
    })

    # Build loser-side stats
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "label": 0,
        "date": df["tourney_date"],
        "week": df["week"],
        "surface": df["surface"],
        "tier_w": df["tier_w"],
        "bp_saved": pd.to_numeric(df["l_bpSaved"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["l_bpFaced"], errors="coerce"),
        "opp_bp_saved": pd.to_numeric(df["w_bpSaved"], errors="coerce"),
        "opp_bp_faced": pd.to_numeric(df["w_bpFaced"], errors="coerce"),
        "score": df["score"]
    })

    out = pd.concat([W, L], ignore_index=True)
    out = out.dropna(subset=["player_id", "opp_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)
    return out

# --------------------------------
# Compute Clutch Index per match
# --------------------------------
def compute_clutch_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ratios
    df["bp_save_rate"] = np.where(df["bp_faced"] > 0, df["bp_saved"] / df["bp_faced"], np.nan)
    df["bp_convert_rate"] = np.where(df["opp_bp_faced"] > 0,
                                     (df["opp_bp_faced"] - df["opp_bp_saved"]) / df["opp_bp_faced"], np.nan)

    # approximate tiebreak rate: count of 7-6 or 6-7 sets in score string
    df["tb_played"] = df["score"].astype(str).str.count("7-6|6-7")
    df["tb_won"] = np.where(df["label"] == 1, df["tb_played"], 0)
    df["tb_rate"] = np.where(df["tb_played"] > 0, df["tb_won"] / df["tb_played"], np.nan)

    # fill missing with neutral 0.5
    for col in ["bp_save_rate", "bp_convert_rate", "tb_rate"]:
        df[col] = df[col].fillna(0.5)

    # compute composite
    df["CPI_adv"] = (
        0.4 * (df["bp_save_rate"] - 0.5)
        + 0.4 * (df["bp_convert_rate"] - 0.5)
        + 0.2 * (df["tb_rate"] - 0.5)
    )

    df["CPI_weighted"] = df["CPI_adv"] * df["tier_w"]

    return df

# --------------------------------
# Weekly aggregation
# --------------------------------
def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["player_id", "player_name", "week"], observed=True)
        .agg(matches=("label", "size"),
             CPI_adv_mean=("CPI_adv", "mean"),
             CPI_weighted_mean=("CPI_weighted", "mean"))
        .reset_index()
    )
    return agg

# --------------------------------
# Main
# --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Clutch Performance Index (Weekly CSV)")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master data: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match dataframe...")
    pm = build_player_match(master)
    print(f"[INFO] Player-match rows: {len(pm):,}")

    print("[INFO] Computing match-level clutch metrics...")
    clutch_match = compute_clutch_match(pm)

    match_out = out_root / "clutch_match.csv"
    clutch_match.to_csv(match_out, index=False)
    print(f"[INFO] Saved clutch match file → {match_out} ({len(clutch_match):,} rows)")

    print("[INFO] Aggregating weekly averages...")
    clutch_weekly = aggregate_weekly(clutch_match)

    weekly_out = out_root / "clutch_weekly.csv"
    clutch_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved clutch weekly file → {weekly_out} ({len(clutch_weekly):,} rows)")

    if not clutch_weekly.empty:
        y = pd.to_datetime(clutch_weekly["week"]).dt.year
        print(f"[INFO] Coverage: {y.min()}–{y.max()} | Players: {clutch_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
