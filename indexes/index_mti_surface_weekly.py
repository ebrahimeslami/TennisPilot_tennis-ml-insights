"""
Mental Toughness Index (MTI) — Weekly + Surface (1991–2025)
============================================================

Outputs
-------
1. mti_weekly.csv           → overall MTI per player-week
2. mti_weekly_surface.csv   → per player-surface-week MTI
3. mti_match.csv (optional) → per-match components (audit)

Usage
-----
python index_mti_surface_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" ^
                                   --out_root "D:\\Tennis\\data\\indexes" ^
                                   --write_match_csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
WEEK_MIN, WEEK_MAX = 1, 52

# ---------------- Helpers ----------------
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

def _tier_weight(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_normalize_tourney_level).map(TIER_WEIGHT).fillna(1.0)

def _iso_year(dt: pd.Series) -> pd.Series:
    return dt.dt.isocalendar().year.astype(int)

def _week_num(dt: pd.Series) -> pd.Series:
    w = dt.dt.isocalendar().week.astype(int)
    return w.clip(lower=WEEK_MIN, upper=WEEK_MAX)

def _week_start(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("W-MON").dt.start_time

def _parse_first_set_games(score_str: str):
    if not isinstance(score_str, str) or len(score_str) == 0:
        return np.nan, np.nan
    s = score_str.strip().split()
    if len(s) == 0:
        return np.nan, np.nan
    token = s[0].split("(")[0]
    parts = token.split("-")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return np.nan, np.nan

def _count_tiebreaks(score_str: str):
    if not isinstance(score_str, str) or len(score_str) == 0:
        return 0, 0
    sets = score_str.strip().split()
    tb_played = 0
    tb_won_by_winner = 0
    for token in sets:
        base = token.split("(")[0]
        if base in ("7-6", "6-7"):
            tb_played += 1
            if base == "7-6":
                tb_won_by_winner += 1
    return tb_played, tb_won_by_winner

def _sets_played(score_str: str) -> int:
    if not isinstance(score_str, str) or len(score_str) == 0:
        return 0
    cnt = 0
    for tok in score_str.strip().split():
        base = tok.split("(")[0]
        if "-" in base:
            a, b = base.split("-", 1)
            if a.isdigit() and b.isdigit():
                cnt += 1
    return cnt

def _expected_win_prob(player_rank, opp_rank, scale=200.0):
    if pd.isna(player_rank) or pd.isna(opp_rank) or player_rank <= 0 or opp_rank <= 0:
        return 0.5
    diff = (1.0 / player_rank) - (1.0 / opp_rank)
    return 1.0 / (1.0 + np.exp(-diff * scale))

# ---------------- Build player-match ----------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    tw = _tier_weight(df.get("tourney_level_norm", df.get("tourney_level", "")))
    surface = df["surface"].fillna("Unknown").str.capitalize()

    date = df["tourney_date"]
    iso_year = _iso_year(date)
    weekn = _week_num(date)
    week_start = _week_start(date)

    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "player_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "label": 1,
        "best_of": pd.to_numeric(df["best_of"], errors="coerce"),
        "score": df["score"].astype(str),
        "bp_saved": pd.to_numeric(df["w_bpSaved"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["w_bpFaced"], errors="coerce"),
        "surface": surface,
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "tier_w": tw,
    })

    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "player_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "label": 0,
        "best_of": pd.to_numeric(df["best_of"], errors="coerce"),
        "score": df["score"].astype(str),
        "bp_saved": pd.to_numeric(df["l_bpSaved"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["l_bpFaced"], errors="coerce"),
        "surface": surface,
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "tier_w": tw,
    })

    PM = pd.concat([W, L], ignore_index=True)
    PM = PM.dropna(subset=["player_id", "opp_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)
    return PM

# ---------------- Match-level component calc ----------------
def compute_match_components(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    fs = pm["score"].apply(_parse_first_set_games)
    pm["fs_w"], pm["fs_l"] = zip(*fs)
    pm["player_fs_games"] = np.where(pm["label"] == 1, pm["fs_w"], pm["fs_l"])
    pm["opp_fs_games"]    = np.where(pm["label"] == 1, pm["fs_l"], pm["fs_w"])
    pm["lost_set1"] = (pm["player_fs_games"] < pm["opp_fs_games"]).astype(int)

    tb_counts = pm["score"].apply(_count_tiebreaks)
    pm["tb_played"], pm["tb_won_by_winner"] = zip(*tb_counts)
    pm["tb_won"] = np.where(pm["label"] == 1, pm["tb_won_by_winner"], pm["tb_played"] - pm["tb_won_by_winner"])

    sets_played = pm["score"].apply(_sets_played)
    pm["sets_played"] = sets_played
    pm["decider_played"] = ((pm["best_of"] >= 3) & (pm["sets_played"] == pm["best_of"])).astype(int)
    pm["decider_win"] = ((pm["decider_played"] == 1) & (pm["label"] == 1)).astype(int)

    pm["bp_saved"] = pd.to_numeric(pm["bp_saved"], errors="coerce").fillna(0.0)
    pm["bp_faced"] = pd.to_numeric(pm["bp_faced"], errors="coerce").fillna(0.0)

    pm["exp_win"] = [_expected_win_prob(pr, ork) for pr, ork in zip(pm["player_rank"], pm["opp_rank"])]
    pm["SF_adj"] = (pm["label"].astype(float) - pm["exp_win"]) * pm["tier_w"]

    return pm

# ---------------- Aggregation (surface or global) ----------------
def aggregate_weekly(df: pd.DataFrame, by_surface=False) -> pd.DataFrame:
    group_cols = ["player_id", "player_name", "iso_year", "week_num", "week_start"]
    if by_surface:
        group_cols.append("surface")

    g = df.groupby(group_cols, observed=True)
    agg = g.agg(
        matches=("label", "size"),
        lost_set1_cnt=("lost_set1", "sum"),
        win_after_lost1=("label", lambda s: int(((s.index.to_series().map(df["lost_set1"]) == 1) & (s == 1)).sum())),
        decider_played=("decider_played", "sum"),
        decider_win=("decider_win", "sum"),
        tb_played=("tb_played", "sum"),
        tb_won=("tb_won", "sum"),
        bp_saved=("bp_saved", "sum"),
        bp_faced=("bp_faced", "sum"),
        SF_mean=("SF_adj", "mean"),
    ).reset_index()

    eps = 1e-9
    agg["SRR"] = np.where(agg["lost_set1_cnt"] > 0, agg["win_after_lost1"] / agg["lost_set1_cnt"], np.nan)
    agg["DWR"] = np.where(agg["decider_played"] > 0, agg["decider_win"] / agg["decider_played"], np.nan)
    agg["TBE"] = np.where(agg["tb_played"] > 0, agg["tb_won"] / agg["tb_played"], np.nan)
    agg["BPR"] = np.where(agg["bp_faced"] >= 5, agg["bp_saved"] / np.maximum(agg["bp_faced"], eps), 0.5)

    for col in ["SRR", "DWR", "TBE"]:
        agg[col] = agg[col].fillna(0.5)
    agg["SF_mean"] = agg["SF_mean"].fillna(0.0)

    agg["MTI"] = (
        0.25 * agg["SRR"] +
        0.25 * agg["DWR"] +
        0.20 * agg["TBE"] +
        0.20 * agg["BPR"] +
        0.10 * agg["SF_mean"]
    )

    return agg.sort_values(group_cols).reset_index(drop=True)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Mental Toughness Index (MTI) — Weekly + Surface CSVs")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--write_match_csv", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match data…")
    pm = build_player_match(master)
    print(f"[INFO] Matches: {len(pm):,}, Players: {pm['player_id'].nunique()}")

    print("[INFO] Computing match-level components…")
    match_df = compute_match_components(pm)

    if args.write_match_csv:
        match_df.to_csv(out_root / "mti_match.csv", index=False)
        print(f"[INFO] Match-level data saved → {out_root / 'mti_match.csv'}")

    print("[INFO] Aggregating weekly MTI (global)…")
    mti_global = aggregate_weekly(match_df, by_surface=False)
    mti_global.to_csv(out_root / "mti_weekly.csv", index=False)
    print(f"[INFO] Global weekly MTI saved → {out_root / 'mti_weekly.csv'} ({len(mti_global):,} rows)")

    print("[INFO] Aggregating weekly MTI (by surface)…")
    mti_surface = aggregate_weekly(match_df, by_surface=True)
    mti_surface.to_csv(out_root / "mti_weekly_surface.csv", index=False)
    print(f"[INFO] Surface weekly MTI saved → {out_root / 'mti_weekly_surface.csv'} ({len(mti_surface):,} rows)")

    print(f"[INFO] Coverage: {int(mti_global['iso_year'].min())}-{int(mti_global['iso_year'].max())}")

if __name__ == "__main__":
    main()
