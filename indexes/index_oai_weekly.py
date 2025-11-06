"""
Opponent Adaptation Index (OAI) — Weekly Series (CSV)
=====================================================

Purpose
-------
Quantify how effectively a player adjusts tactics versus their recent baseline
and whether that adjustment aligns with an in-match turnaround.

Key ideas
---------
1) Direction (adapt_sign): inferred from the first set result vs final result.
   +1 if lost Set 1 but won the match (positive adaptation)
   -1 if won Set 1 but lost the match (negative adaptation)
    0 if straight-sets or ambiguous/no first set info (neutral)

2) Magnitude (shift_mag): absolute deviation of this match's key performance
   features from the player's own rolling baseline on the same surface:
     - first-serve in %
     - first-serve points won %
     - second-serve points won %
     - return points won %

   shift_mag = mean( |feature_match - feature_baseline| )

3) OAI_raw = shift_mag × adapt_sign
   OAI_weighted = OAI_raw × TierWeight × OppStrengthWeight

OppStrengthWeight = 1 / log1p(opp_rank), clipped to [0.5, 2.0]

Outputs
-------
- oai_match.csv  : one row per player-match with OAI metrics
- oai_weekly.csv : weekly average OAI per player

Usage
-----
python index_oai_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes" --baseline_window 10
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# -----------------------------
# Config / constants
# -----------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

FEATURE_NAMES = ["fs_in", "fs_win", "ss_win", "ret_pts_won"]  # used for shift magnitude

# -----------------------------
# Helpers
# -----------------------------
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

def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").fillna(0).astype(float)
    b = pd.to_numeric(b, errors="coerce").fillna(0).astype(float)
    out = np.zeros_like(a, dtype=float)
    mask = b != 0
    out[mask] = (a[mask] / b[mask])
    return np.clip(out, 0.0, 1.0)

def _parse_first_set_outcome(score_str: str, player_label: int) -> int:
    """
    Return +1 if player won Set 1, -1 if lost Set 1, 0 if unknown/ambiguous.
    player_label: 1 if this row is winner side, 0 if loser side.
    Handles common formats: "6-4 4-6 6-3", "7-6(5) 6-3", "RET", "W/O", etc.
    """
    if not isinstance(score_str, str) or score_str.strip() == "":
        return 0
    s = score_str.upper()
    # Exclude non-standard finishes where adaptation is ambiguous
    if any(tag in s for tag in ["RET", "W/O", "DEF", "ABN"]):
        return 0
    # take first set token
    first = s.split()[0] if " " in s else s
    # strip tiebreaks like 7-6(5)
    m = re.match(r"^\s*(\d{1,2})-(\d{1,2})", first)
    if not m:
        return 0
    g1, g2 = int(m.group(1)), int(m.group(2))
    if player_label == 1:  # this row is winner side
        return 1 if g1 > g2 else -1
    else:  # loser side row
        return 1 if g1 < g2 else -1

def _opponent_strength_weight(rank: float) -> float:
    """
    Convert opponent rank to a weight in [0.5, 2.0].
    Lower rank (stronger opponent) → higher weight.
    """
    if pd.isna(rank) or rank <= 0:
        return 1.0
    w = 1.0 / np.log1p(rank)
    return float(np.clip(w, 0.5, 2.0))


# -----------------------------
# Build per-player-match rows with features
# -----------------------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    """Winner/loser → player rows with core features and context."""
    df = df.copy()
    # robust date
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # tier
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)
    df["tier_w"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    # common
    date = df["tourney_date"]
    week = _to_week(date)
    surface = df["surface"].fillna("Hard").astype(str)
    score = df["score"].astype(str)

    # ------- Winner side rows -------
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "player_rank": pd.to_numeric(df.get("winner_rank"), errors="coerce"),
        "opp_rank": pd.to_numeric(df.get("loser_rank"), errors="coerce"),
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
        "label": 1,
        "score": score,
        # features (serve)
        "fs_in": _safe_div(df["w_1stIn"], df["w_svpt"]),
        "fs_win": _safe_div(df["w_1stWon"], df["w_1stIn"]),
        "ss_win": _safe_div(df["w_2ndWon"], (pd.to_numeric(df["w_svpt"], errors="coerce") - pd.to_numeric(df["w_1stIn"], errors="coerce"))),
        # features (return) via opp serve
        "ret_pts_won": _safe_div(
            (pd.to_numeric(df.get("l_svpt"), errors="coerce") - pd.to_numeric(df.get("l_1stWon"), errors="coerce") - pd.to_numeric(df.get("l_2ndWon"), errors="coerce")),
            pd.to_numeric(df.get("l_svpt"), errors="coerce")
        ),
    })

    # ------- Loser side rows -------
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "player_rank": pd.to_numeric(df.get("loser_rank"), errors="coerce"),
        "opp_rank": pd.to_numeric(df.get("winner_rank"), errors="coerce"),
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
        "label": 0,
        "score": score,
        "fs_in": _safe_div(df["l_1stIn"], df["l_svpt"]),
        "fs_win": _safe_div(df["l_1stWon"], df["l_1stIn"]),
        "ss_win": _safe_div(df["l_2ndWon"], (pd.to_numeric(df["l_svpt"], errors="coerce") - pd.to_numeric(df["l_1stIn"], errors="coerce"))),
        "ret_pts_won": _safe_div(
            (pd.to_numeric(df.get("w_svpt"), errors="coerce") - pd.to_numeric(df.get("w_1stWon"), errors="coerce") - pd.to_numeric(df.get("w_2ndWon"), errors="coerce")),
            pd.to_numeric(df.get("w_svpt"), errors="coerce")
        ),
    })

    out = pd.concat([W, L], ignore_index=True)
    # keep bounded
    for col in FEATURE_NAMES:
        out[col] = out[col].astype(float).clip(0.0, 1.0)

    # first set outcome (+1/-1/0)
    out["first_set_result"] = [
        _parse_first_set_outcome(s, lab) for s, lab in zip(out["score"].astype(str), out["label"].astype(int))
    ]
    # adapt sign: compare first set vs final match result
    out["adapt_sign"] = 0
    # lost set1 but won match => +1 ; won set1 but lost match => -1
    mask_pos = (out["first_set_result"] == -1) & (out["label"] == 1)
    mask_neg = (out["first_set_result"] == 1) & (out["label"] == 0)
    out.loc[mask_pos, "adapt_sign"] = 1
    out.loc[mask_neg, "adapt_sign"] = -1

    # opponent strength weight
    out["opp_w"] = out["opp_rank"].apply(_opponent_strength_weight).astype(float)
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


# -----------------------------
# Compute rolling baselines & OAI per match
# -----------------------------
def compute_oai_per_match(df_pm: pd.DataFrame, baseline_window: int = 10) -> pd.DataFrame:
    """
    For each player, same-surface rolling baselines over previous `baseline_window` matches.
    OAI_raw = mean(|feat - baseline_feat|) * adapt_sign
    OAI_weighted = OAI_raw * tier_w * opp_w
    """
    rows = []
    # group by player & surface for surface-specific baselines
    for (pid, surf), g in df_pm.groupby(["player_id", "surface"]):
        g = g.sort_values("date").reset_index(drop=True)

        # build rolling baselines (previous matches only -> shift)
        for feat in FEATURE_NAMES:
            g[f"base_{feat}"] = (
                g[feat].shift(1).rolling(window=baseline_window, min_periods=3).mean()
            )

        # compute shift magnitude
        diffs = []
        for feat in FEATURE_NAMES:
            diffs.append((g[feat] - g[f"base_{feat}"]).abs())
        g["shift_mag"] = pd.concat(diffs, axis=1).mean(axis=1)

        # raw and weighted OAI
        g["OAI_raw"] = g["shift_mag"] * g["adapt_sign"]
        g["OAI_weighted"] = g["OAI_raw"] * g["tier_w"].fillna(1.0) * g["opp_w"].fillna(1.0)

        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    return out


# -----------------------------
# Weekly aggregation
# -----------------------------
def weekly_oai(df_match_oai: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly average OAI per player (weighted and raw), plus counts.
    """
    grp = (
        df_match_oai.groupby(["player_id", "player_name", "week"], observed=True)
        .agg(
            OAI_raw_mean=("OAI_raw", "mean"),
            OAI_weighted_mean=("OAI_weighted", "mean"),
            matches=("OAI_raw", "size"),
            mean_shift_mag=("shift_mag", "mean"),
        )
        .reset_index()
        .sort_values(["player_id", "week"])
    )
    # standardize column names
    grp = grp.rename(columns={
        "OAI_raw_mean": "OAI_raw_weekly",
        "OAI_weighted_mean": "OAI_weighted_weekly"
    })
    return grp


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Opponent Adaptation Index (OAI) — Weekly CSV")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--baseline_window", type=int, default=10, help="Rolling window for baselines per surface")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building per-player match table...")
    pm = build_player_match(master)
    print(f"[INFO] Player-match rows: {len(pm):,} for {pm['player_id'].nunique():,} players")

    print("[INFO] Computing OAI per match with rolling baselines...")
    oai_match = compute_oai_per_match(pm, baseline_window=args.baseline_window)

    match_out = out_root / "oai_match.csv"
    cols_out = [
        "player_id","player_name","opp_id","opp_name","date","week","surface","tier_w","label",
        "player_rank","opp_rank","score","first_set_result","adapt_sign",
        "fs_in","fs_win","ss_win","ret_pts_won",
        "base_fs_in","base_fs_win","base_ss_win","base_ret_pts_won",
        "shift_mag","opp_w","OAI_raw","OAI_weighted"
    ]
    # Ensure baseline columns exist (naming from loop)
    for feat in FEATURE_NAMES:
        base_col = f"base_{feat}"
        if base_col not in oai_match.columns:
            oai_match[base_col] = np.nan

    oai_match[cols_out].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level OAI → {match_out} ({len(oai_match):,} rows)")

    print("[INFO] Aggregating to weekly OAI...")
    oai_weekly = weekly_oai(oai_match)

    weekly_out = out_root / "oai_weekly.csv"
    oai_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly OAI → {weekly_out} ({len(oai_weekly):,} rows)")

    if not oai_weekly.empty:
        years = pd.to_datetime(oai_weekly["week"]).dt.year
        print(f"[INFO] Coverage years: {years.min()}–{years.max()} | Players: {oai_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
