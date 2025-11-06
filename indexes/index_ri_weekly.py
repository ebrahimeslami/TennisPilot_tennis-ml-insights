# -*- coding: utf-8 -*-
"""
Resilience Index (RI)
---------------------
Measures within-match recovery ability for each player using three components:
  1) Set Recovery (Rs): chance to win the match after losing the first set
  2) Break Recovery (Rb): breaks-won / breaks-conceded in the match
  3) Pressure Recovery (Rp): deciding-set win proxy (1 if win when match goes to decider)

Each component is normalized by the player's own prior 52-week baseline
(time-based rolling window, shifted to avoid lookahead). RI_match is a
weighted average of the available normalized components; weights are
re-normalized if some components are not applicable in a given match.

Outputs:
  - ri_match.csv            (player–match rows with components and RI_match)
  - ri_weekly.csv           (player–week average RI)
  - ri_weekly_surface.csv   (player–week–surface average RI)
"""

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------
# Basic helpers
# ---------------------------
SET_PAT = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)")
TB_OR_WO = {"RET", "W/O", "WO", "DEF", "ABN"}

def to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    """Convert int-like YYYYMMDD to pandas datetime (NaT on errors)."""
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s: object) -> str:
    if not isinstance(s, str):
        return "Other"
    t = s.strip().lower()
    if "hard" in t:  return "Hard"
    if "clay" in t:  return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t or "indoor" in t: return "Carpet"
    return "Other"

def parse_first_set_tuple(score: object):
    """Return (w_games, l_games) for the first set in winner-oriented score string."""
    if not isinstance(score, str) or not score.strip():
        return (np.nan, np.nan)
    tokens = [tok for tok in score.strip().split() if tok.upper() not in TB_OR_WO]
    if not tokens:
        return (np.nan, np.nan)
    m = SET_PAT.match(tokens[0])
    if not m:
        return (np.nan, np.nan)
    return int(m.group(1)), int(m.group(2))

def decider_flag(score: object, best_of: object) -> int:
    """1 if match went to deciding set (3rd or 5th), else 0."""
    if not isinstance(score, str) or pd.isna(best_of):
        return 0
    toks = [t for t in score.strip().split() if t.upper() not in TB_OR_WO]
    if not toks:
        return 0
    nsets = len(toks)
    max_sets = 5 if int(best_of) >= 5 else 3
    return int(nsets == max_sets)


# ---------------------------
# Build long player–match table
# ---------------------------
def build_long(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991].copy()

    df["surface_c"] = df["surface"].map(canon_surface)
    df["iso_year"]   = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"]   = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    # First set tuple from winner perspective
    fs_w, fs_l = zip(*df["score"].map(parse_first_set_tuple))
    df["fs_w"] = pd.to_numeric(fs_w, errors="coerce")
    df["fs_l"] = pd.to_numeric(fs_l, errors="coerce")

    df["decider"] = [decider_flag(s, b) for s, b in zip(df["score"], df["best_of"])]

    # Winner view
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "surface": df["surface_c"],
        "tourney_id": df["tourney_id"],
        "round": df["round"],
        "best_of": df["best_of"],
        "minutes": df["minutes"],
        "label": 1,  # match won
        # first set result from THIS player's perspective (winner)
        "lost_first_set": (df["fs_w"] < df["fs_l"]).astype("Int64"),
        # serve/return stats needed for breaks
        "bp_faced": df["w_bpFaced"],
        "bp_saved": df["w_bpSaved"],
        "opp_bp_faced": df["l_bpFaced"],
        "opp_bp_saved": df["l_bpSaved"],
        "decider": df["decider"].astype(int),
    })

    # Loser view
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "surface": df["surface_c"],
        "tourney_id": df["tourney_id"],
        "round": df["round"],
        "best_of": df["best_of"],
        "minutes": df["minutes"],
        "label": 0,  # match lost
        # for loser, they lost first set if winner won first set (fs_w > fs_l)
        "lost_first_set": (df["fs_w"] > df["fs_l"]).astype("Int64"),
        "bp_faced": df["l_bpFaced"],
        "bp_saved": df["l_bpSaved"],
        "opp_bp_faced": df["w_bpFaced"],
        "opp_bp_saved": df["w_bpSaved"],
        "decider": df["decider"].astype(int),
    })

    PM = pd.concat([W, L], ignore_index=True)
    # compute break components for the player perspective
    PM["breaks_conceded"] = (PM["bp_faced"] - PM["bp_saved"]).clip(lower=0)
    PM["breaks_won"]      = (PM["opp_bp_faced"] - PM["opp_bp_saved"]).clip(lower=0)
    # per-match Break Recovery ratio (bounded to [0,1] for stability)
    PM["Rb_raw"] = np.where(PM["breaks_conceded"] > 0,
                            PM["breaks_won"] / PM["breaks_conceded"],
                            np.where(PM["breaks_won"].notna(), 1.0, np.nan))
    PM["Rb_raw"] = PM["Rb_raw"].clip(lower=0, upper=1)

    # Set recovery indicator only when lost_first_set == 1
    PM["Rs_raw"] = np.where(PM["lost_first_set"] == 1, PM["label"].astype(float), np.nan)

    # Pressure recovery proxy: deciding-set win indicator
    PM["Rp_raw"] = np.where(PM["decider"] == 1, PM["label"].astype(float), np.nan)

    PM = PM.sort_values(["player_id", "date", "tourney_id"]).reset_index(drop=True)
    return PM


# ---------------------------
# Baselines (prior 52 weeks)
# ---------------------------
def add_prior_52w_baselines(PM: pd.DataFrame) -> pd.DataFrame:
    """Time-based rolling means (365D) shifted by 1 to avoid lookahead."""
    out_list = []
    for pid, sub in PM.groupby("player_id", sort=False):
        s = sub.sort_values("date").copy()
        s = s.set_index("date")

        # shift to ensure prior-only information
        for col in ["Rs_raw", "Rb_raw", "Rp_raw"]:
            s[col + "_prior"] = s[col].shift(1)

        # time-based rolling window over 365 days
        s["Rs_base_52w"] = s["Rs_raw_prior"].rolling("365D", min_periods=5).mean()
        s["Rb_base_52w"] = s["Rb_raw_prior"].rolling("365D", min_periods=8).mean()
        s["Rp_base_52w"] = s["Rp_raw_prior"].rolling("365D", min_periods=5).mean()

        # fill conservative defaults where no baseline exists
        s["Rs_base_52w"] = s["Rs_base_52w"].fillna(0.30)  # typical comeback rate after losing set1
        s["Rb_base_52w"] = s["Rb_base_52w"].fillna(0.50)
        s["Rp_base_52w"] = s["Rp_base_52w"].fillna(0.50)

        out_list.append(s.reset_index())

    PMb = pd.concat(out_list, ignore_index=True)
    return PMb


# ---------------------------
# Compute normalized components & RI
# ---------------------------
def compute_ri_match(PMb: pd.DataFrame,
                     w_s: float = 0.40,
                     w_b: float = 0.30,
                     w_p: float = 0.30) -> pd.DataFrame:
    df = PMb.copy()

    # Normalized deltas (component - baseline)
    df["Rs_norm"] = df["Rs_raw"] - df["Rs_base_52w"]
    df["Rb_norm"] = df["Rb_raw"] - df["Rb_base_52w"]
    df["Rp_norm"] = df["Rp_raw"] - df["Rp_base_52w"]

    # For each match, reweight by available components only
    weights = np.vstack([
        (~df["Rs_norm"].isna()).astype(float) * w_s,
        (~df["Rb_norm"].isna()).astype(float) * w_b,
        (~df["Rp_norm"].isna()).astype(float) * w_p
    ]).T  # shape (N, 3)

    wsum = weights.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)

    comp = np.vstack([
        df["Rs_norm"].fillna(0.0),
        df["Rb_norm"].fillna(0.0),
        df["Rp_norm"].fillna(0.0)
    ]).T  # shape (N, 3)

    df["RI_match"] = (weights * comp).sum(axis=1) / wsum

    return df


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
            RI_mean=("RI_match","mean"),
            Rs_mean=("Rs_norm","mean"),
            Rb_mean=("Rb_norm","mean"),
            Rp_mean=("Rp_norm","mean"),
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
    ap = argparse.ArgumentParser(description="Resilience Index (RI): match / weekly / weekly_surface")
    ap.add_argument("--master", required=True, help="Path to master parquet (e.g., D:\\Tennis\\data\\master\\tennis_master_1991.parquet)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    # optional custom weights
    ap.add_argument("--w_s", type=float, default=0.40, help="Weight for Set Recovery (default 0.40)")
    ap.add_argument("--w_b", type=float, default=0.30, help="Weight for Break Recovery (default 0.30)")
    ap.add_argument("--w_p", type=float, default=0.30, help="Weight for Pressure Recovery (default 0.30)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player–match table…")
    PM = build_long(master)

    print("[INFO] Adding prior 52-week baselines…")
    PMb = add_prior_52w_baselines(PM)

    print("[INFO] Computing match-level RI…")
    PMc = compute_ri_match(PMb, w_s=args.w_s, w_b=args.w_b, w_p=args.w_p)

    # Match-level output
    match_cols = [
        "player_id","player_name","opp_id","opp_name",
        "date","iso_year","week_num","week_start","surface",
        "tourney_id","round","best_of","minutes",
        "label","lost_first_set","decider",
        "breaks_won","breaks_conceded",
        "Rs_raw","Rb_raw","Rp_raw",
        "Rs_base_52w","Rb_base_52w","Rp_base_52w",
        "Rs_norm","Rb_norm","Rp_norm",
        "RI_match"
    ]
    match_out = out_root / "ri_match.csv"
    PMc[match_cols].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level RI → {match_out} ({len(PMc):,} rows)")

    # Weekly global
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(PMc, by_surface=False)
    weekly_out = out_root / "ri_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly RI → {weekly_out} ({len(weekly):,} rows)")

    # Weekly by surface
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(PMc, by_surface=True)
    weekly_surf_out = out_root / "ri_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface RI → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
