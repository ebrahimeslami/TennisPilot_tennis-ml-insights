"""
Consistency Stability Index (CSI) — Weekly Series (CSV)
=======================================================

Purpose
--------
Quantifies how stable and consistent a player's technical performance is over time.
A high CSI means a player maintains stable serve and return metrics across matches,
while a low CSI indicates large fluctuations in execution (volatile performance).

Definition
----------
For each player and surface:
    CSI_raw = 1 - normalized_std(mean([fs_in, fs_win, ss_win, ret_pts_won]))
    CSI_weighted = CSI_raw * TierWeight * OppStrengthWeight

Rolling window (default=10 matches) smooths the short-term variation.

Outputs
--------
- csi_match.csv  : CSI per player per match (raw + weighted)
- csi_weekly.csv : weekly average CSI (raw + weighted) per player

Usage
------
python index_csi_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes" --window 10
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
# Compute CSI
# ----------------------------
def compute_csi_per_match(df_pm: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    For each player & surface, compute rolling std deviation across last N matches.
    CSI_raw = 1 - normalized_std(mean([fs_in, fs_win, ss_win, ret_pts_won]))
    """
    all_rows = []
    for (pid, surf), g in df_pm.groupby(["player_id", "surface"]):
        g = g.sort_values("date").reset_index(drop=True)

        # rolling std per feature
        for feat in FEATURE_NAMES:
            g[f"std_{feat}"] = (
                g[feat].rolling(window=window, min_periods=3).std()
            )

        g["std_mean"] = g[[f"std_{f}" for f in FEATURE_NAMES]].mean(axis=1)
        # normalize and invert (higher = more consistent)
        g["CSI_raw"] = 1 - g["std_mean"].fillna(0)
        g["CSI_weighted"] = g["CSI_raw"] * g["tier_w"] * g["opp_w"]

        all_rows.append(g)
    return pd.concat(all_rows, ignore_index=True)

# ----------------------------
# Weekly aggregation
# ----------------------------
def weekly_csi(df_match_csi: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df_match_csi.groupby(["player_id", "player_name", "week"], observed=True)
        .agg(
            CSI_raw_weekly=("CSI_raw", "mean"),
            CSI_weighted_weekly=("CSI_weighted", "mean"),
            matches=("CSI_raw", "size"),
        )
        .reset_index()
        .sort_values(["player_id", "week"])
    )
    return grp

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Consistency Stability Index (CSI) — Weekly CSV")
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

    print("[INFO] Computing rolling CSI...")
    csi_match = compute_csi_per_match(pm, window=args.window)

    match_out = out_root / "csi_match.csv"
    csi_match.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level CSI → {match_out} ({len(csi_match):,} rows)")

    print("[INFO] Aggregating weekly CSI...")
    csi_weekly = weekly_csi(csi_match)

    weekly_out = out_root / "csi_weekly.csv"
    csi_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly CSI → {weekly_out} ({len(csi_weekly):,} rows)")

    if not csi_weekly.empty:
        yrs = pd.to_datetime(csi_weekly["week"]).dt.year
        print(f"[INFO] Coverage years: {yrs.min()}–{yrs.max()} | Players: {csi_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
