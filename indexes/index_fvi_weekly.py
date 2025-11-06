"""
Form Volatility Index (FVI) — Weekly Series (CSV)
=================================================

Purpose
--------
Quantifies how streaky or unpredictable a player's performance form is over time.

Definition
----------
FVI_raw = rolling_std(PerfScore) / (rolling_mean(PerfScore) + eps)
PerfScore = mean(fs_in, fs_win, ss_win, ret_pts_won)

Weighted by tournament tier and opponent strength:
FVI_weighted = FVI_raw * TierWeight * OppStrengthWeight

Outputs
--------
- fvi_match.csv  : FVI per player per match (raw + weighted)
- fvi_weekly.csv : weekly average FVI per player

Usage
------
python index_fvi_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes" --window 10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Global constants
# ----------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
FEATURE_NAMES = ["fs_in", "fs_win", "ss_win", "ret_pts_won"]

# ----------------------------
# Helper functions
# ----------------------------
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

def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").fillna(0)
    b = pd.to_numeric(b, errors="coerce").fillna(0)
    out = np.zeros_like(a, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return np.clip(out, 0, 1)

def _to_week(dt: pd.Series):
    return dt.dt.to_period("W-MON").dt.start_time

def _opp_weight(rank: float) -> float:
    """Convert opponent rank to strength weight."""
    if pd.isna(rank) or rank <= 0:
        return 1.0
    w = 1.0 / np.log1p(rank)
    return float(np.clip(w, 0.5, 2.0))

# ----------------------------
# Build player-match data
# ----------------------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    df["tier_norm"] = df.get("tourney_level_norm", df.get("tourney_level", "")).astype(str).map(_normalize_tourney_level)
    df["tier_w"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    date = df["tourney_date"]
    week = _to_week(date)
    surface = df["surface"].fillna("Hard")

    # Winner side
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "player_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
        "label": 1,
        "fs_in": _safe_div(df["w_1stIn"], df["w_svpt"]),
        "fs_win": _safe_div(df["w_1stWon"], df["w_1stIn"]),
        "ss_win": _safe_div(df["w_2ndWon"], (df["w_svpt"] - df["w_1stIn"])),
        "ret_pts_won": _safe_div(
            (df["l_svpt"] - df["l_1stWon"] - df["l_2ndWon"]), df["l_svpt"]
        ),
    })

    # Loser side
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "player_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
        "label": 0,
        "fs_in": _safe_div(df["l_1stIn"], df["l_svpt"]),
        "fs_win": _safe_div(df["l_1stWon"], df["l_1stIn"]),
        "ss_win": _safe_div(df["l_2ndWon"], (df["l_svpt"] - df["l_1stIn"])),
        "ret_pts_won": _safe_div(
            (df["w_svpt"] - df["w_1stWon"] - df["w_2ndWon"]), df["w_svpt"]
        ),
    })

    out = pd.concat([W, L], ignore_index=True)
    for c in FEATURE_NAMES:
        out[c] = out[c].astype(float).clip(0.0, 1.0)
    out["opp_w"] = out["opp_rank"].apply(_opp_weight)
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)

# ----------------------------
# Compute FVI
# ----------------------------
def compute_fvi_per_match(df_pm: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    For each player & surface:
      PerfScore = mean(fs_in, fs_win, ss_win, ret_pts_won)
      FVI_raw = rolling_std(PerfScore) / (rolling_mean(PerfScore) + eps)
    """
    rows = []
    eps = 1e-6
    for (pid, surf), g in df_pm.groupby(["player_id", "surface"]):
        g = g.sort_values("date").reset_index(drop=True)
        g["PerfScore"] = g[FEATURE_NAMES].mean(axis=1)

        g["mean_perf"] = g["PerfScore"].rolling(window=window, min_periods=3).mean()
        g["std_perf"] = g["PerfScore"].rolling(window=window, min_periods=3).std()

        g["FVI_raw"] = g["std_perf"] / (g["mean_perf"] + eps)
        g["FVI_weighted"] = g["FVI_raw"] * g["tier_w"] * g["opp_w"]

        rows.append(g)
    return pd.concat(rows, ignore_index=True)

# ----------------------------
# Weekly aggregation
# ----------------------------
def weekly_fvi(df_match_fvi: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df_match_fvi.groupby(["player_id", "player_name", "week"], observed=True)
        .agg(
            FVI_raw_weekly=("FVI_raw", "mean"),
            FVI_weighted_weekly=("FVI_weighted", "mean"),
            matches=("FVI_raw", "size"),
            mean_perf=("PerfScore", "mean"),
        )
        .reset_index()
        .sort_values(["player_id", "week"])
    )
    return grp

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Form Volatility Index (FVI) — Weekly CSV")
    ap.add_argument("--master", required=True, help="Path to master parquet")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--window", type=int, default=10, help="Rolling window in matches")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building per-player match dataset...")
    pm = build_player_match(master)
    print(f"[INFO] {len(pm):,} player-match rows loaded.")

    print("[INFO] Computing rolling FVI...")
    fvi_match = compute_fvi_per_match(pm, window=args.window)

    match_out = out_root / "fvi_match.csv"
    fvi_match.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level FVI → {match_out} ({len(fvi_match):,} rows)")

    print("[INFO] Aggregating weekly FVI...")
    fvi_weekly = weekly_fvi(fvi_match)

    weekly_out = out_root / "fvi_weekly.csv"
    fvi_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly FVI → {weekly_out} ({len(fvi_weekly):,} rows)")

    if not fvi_weekly.empty:
        yrs = pd.to_datetime(fvi_weekly["week"]).dt.year
        print(f"[INFO] Coverage years: {yrs.min()}–{yrs.max()} | Players: {fvi_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
