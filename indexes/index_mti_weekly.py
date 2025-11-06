"""
Mental Toughness Index (MTI) — Weekly (1991–2025)
=================================================

What it measures
----------------
Composite pressure resilience per player-week:
- SRR (Set Recovery Rate): win after losing set 1
- DWR (Decider Win Rate): win deciding set (3rd/5th)
- TBE (Tiebreak Efficiency): tiebreaks won/played
- BPR (Break-Point Resilience): BP_saved/BP_faced (robust)
- SF (Surprise Factor): (actual - expected) * tier_weight

Weekly MTI:
MTI = 0.25*SRR + 0.25*DWR + 0.20*TBE + 0.20*BPR + 0.10*SF_mean

Outputs
-------
- mti_weekly.csv : player_id, player_name, iso_year, week_num, week_start, SRR, DWR, TBE, BPR, SF_mean, MTI, matches
- mti_match.csv  : (optional) per-match components for debugging

Usage
-----
python index_mti_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes"
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
    """
    Returns (w1, l1) integer games of set 1 from a Tennis Abstract style score string.
    Handles patterns like '4-6 6-3 6-4', '7-6(5) 6-7(4) 7-6(8)', 'RET', 'W/O'.
    If parsing fails, returns (np.nan, np.nan).
    """
    if not isinstance(score_str, str) or len(score_str) == 0:
        return np.nan, np.nan
    s = score_str.strip().split()
    if len(s) == 0:
        return np.nan, np.nan
    token = s[0]
    # Remove tiebreak parentheses e.g. 7-6(5)
    token = token.split("(")[0]
    parts = token.split("-")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        w1 = int(parts[0])
        l1 = int(parts[1])
        return w1, l1
    except Exception:
        return np.nan, np.nan

def _count_tiebreaks(score_str: str):
    """Return (tb_played, tb_won_by_winner) from score string."""
    if not isinstance(score_str, str) or len(score_str) == 0:
        return 0, 0
    # Count occurrences of 7-6 or 6-7 (winner-perspective sets)
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
    # Count only valid set tokens like '6-4', '7-6(5)', ignore RET/W/O
    cnt = 0
    for tok in score_str.strip().split():
        base = tok.split("(")[0]
        if "-" in base:
            a, b = base.split("-", 1)
            if a.isdigit() and b.isdigit():
                cnt += 1
    return cnt

def _expected_win_prob(player_rank, opp_rank, scale=200.0):
    """Smooth logistic expectation from rank differential.
    Higher opp_rank (worse opponent) => higher expected P(win).
    """
    pr = np.nan if pd.isna(player_rank) else float(player_rank)
    ork = np.nan if pd.isna(opp_rank) else float(opp_rank)
    if np.isnan(pr) or np.isnan(ork) or pr <= 0 or ork <= 0:
        return 0.5
    # Positive diff => underdog (opp better rank number is smaller)
    # We invert: lower rank number is stronger. Use (player_strength - opp_strength) approx as (1/pr - 1/ork)
    diff = (1.0 / pr) - (1.0 / ork)
    # Map diff into logistic
    return 1.0 / (1.0 + np.exp(-diff * scale))

# ------------- Build player-match rows -------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # dates
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # tier weight
    tier_raw = df.get("tourney_level_norm", df.get("tourney_level", ""))
    tw = _tier_weight(pd.Series(tier_raw, index=df.index))

    # shared date parts
    date = df["tourney_date"]
    iso_year = _iso_year(date)
    weekn = _week_num(date)
    week_start = _week_start(date)

    # winner-centric
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
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "tier_w": tw,
    })

    # loser-centric
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
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "tier_w": tw,
    })

    PM = pd.concat([W, L], ignore_index=True)
    PM = PM.dropna(subset=["player_id", "opp_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)
    return PM

# ------------- Compute per-match components -------------
def compute_match_components(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()

    # First-set result (player perspective)
    fs = pm["score"].apply(_parse_first_set_games)
    pm["fs_w"], pm["fs_l"] = zip(*fs)
    pm["player_fs_games"] = np.where(pm["label"] == 1, pm["fs_w"], pm["fs_l"])
    pm["opp_fs_games"]    = np.where(pm["label"] == 1, pm["fs_l"], pm["fs_w"])
    pm["lost_set1"] = (pm["player_fs_games"] < pm["opp_fs_games"]).astype(int)

    # Tiebreaks
    tb_counts = pm["score"].apply(_count_tiebreaks)
    pm["tb_played"], pm["tb_won_by_winner"] = zip(*tb_counts)
    # Player tiebreaks won (winner rows get tb_won_by_winner; loser rows get (played - winner_won))
    pm["tb_won"] = np.where(pm["label"] == 1, pm["tb_won_by_winner"], pm["tb_played"] - pm["tb_won_by_winner"])

    # Decider played and decider win
    sets_played = pm["score"].apply(_sets_played)
    pm["sets_played"] = sets_played
    pm["decider_played"] = ((pm["best_of"] >= 3) & (pm["sets_played"] == pm["best_of"])).astype(int)
    pm["decider_win"] = ((pm["decider_played"] == 1) & (pm["label"] == 1)).astype(int)

    # BPR (robust)
    pm["bp_saved"] = pd.to_numeric(pm["bp_saved"], errors="coerce").fillna(0.0)
    pm["bp_faced"] = pd.to_numeric(pm["bp_faced"], errors="coerce").fillna(0.0)

    # Surprise factor (expected win from ranks)
    pm["exp_win"] = [_expected_win_prob(pr, ork) for pr, ork in zip(pm["player_rank"], pm["opp_rank"])]
    pm["SF_adj"] = (pm["label"].astype(float) - pm["exp_win"]) * pm["tier_w"]

    return pm

# ------------- Weekly aggregation -------------
def aggregate_weekly(match_df: pd.DataFrame) -> pd.DataFrame:
    # Per player-year-week aggregates
    g = match_df.groupby(["player_id", "player_name", "iso_year", "week_num", "week_start"], observed=True)

    agg = g.agg(
        matches=("label", "size"),
        lost_set1_cnt=("lost_set1", "sum"),
        win_after_lost1=("label", lambda s: int(((s.index.to_series().map(match_df["lost_set1"]) == 1) & (s == 1)).sum())),
        decider_played=("decider_played", "sum"),
        decider_win=("decider_win", "sum"),
        tb_played=("tb_played", "sum"),
        tb_won=("tb_won", "sum"),
        bp_saved=("bp_saved", "sum"),
        bp_faced=("bp_faced", "sum"),
        SF_mean=("SF_adj", "mean"),
    ).reset_index()

    # Rates with safe denominators
    eps = 1e-9
    agg["SRR"] = np.where(agg["lost_set1_cnt"] > 0, agg["win_after_lost1"] / agg["lost_set1_cnt"], np.nan)
    agg["DWR"] = np.where(agg["decider_played"] > 0, agg["decider_win"] / agg["decider_played"], np.nan)
    agg["TBE"] = np.where(agg["tb_played"] > 0, agg["tb_won"] / agg["tb_played"], np.nan)

    # BPR: if small faced, use neutral 0.5
    agg["BPR"] = np.where(agg["bp_faced"] >= 5, agg["bp_saved"] / np.maximum(agg["bp_faced"], eps), 0.5)

    # Fill missing with neutral 0.5 (for SRR/DWR/TBE where not observed this week)
    for col in ["SRR", "DWR", "TBE"]:
        agg[col] = agg[col].fillna(0.5)

    # SF_mean default 0 if no matches in that cell
    agg["SF_mean"] = agg["SF_mean"].fillna(0.0)

    # Composite MTI
    agg["MTI"] = (
        0.25 * agg["SRR"] +
        0.25 * agg["DWR"] +
        0.20 * agg["TBE"] +
        0.20 * agg["BPR"] +
        0.10 * agg["SF_mean"]
    )

    # Order columns
    cols = [
        "player_id","player_name","iso_year","week_num","week_start",
        "matches",
        "SRR","DWR","TBE","BPR","SF_mean","MTI",
        "lost_set1_cnt","win_after_lost1","decider_played","decider_win","tb_played","tb_won","bp_saved","bp_faced"
    ]
    agg = agg[cols].sort_values(["player_id","iso_year","week_num"]).reset_index(drop=True)
    return agg

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Mental Toughness Index (MTI) — Weekly CSV")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory")
    ap.add_argument("--write_match_csv", action="store_true", help="Also save mti_match.csv for auditing")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match table…")
    pm = build_player_match(master)
    print(f"[INFO] Player-match rows: {len(pm):,}")

    print("[INFO] Computing per-match components…")
    match_df = compute_match_components(pm)

    if args.write_match_csv:
        match_out = out_root / "mti_match.csv"
        match_df.to_csv(match_out, index=False)
        print(f"[INFO] Saved match-level components → {match_out} ({len(match_df):,} rows)")

    print("[INFO] Aggregating weekly MTI…")
    weekly = aggregate_weekly(match_df)

    weekly_out = out_root / "mti_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly MTI → {weekly_out} ({len(weekly):,} rows)")

    if not weekly.empty:
        years = weekly["iso_year"]
        print(f"[INFO] Coverage: {int(years.min())}–{int(years.max())} | Players: {weekly['player_id'].nunique()}")

if __name__ == "__main__":
    main()
