# -*- coding: utf-8 -*-
"""
Pressure Handling Index (PHI)
-----------------------------
Blends four pressure subdomains, each baseline-normalized (prior 52w):
  - BSQ: Break-Point Save Quality (bp_saved / bp_faced)
  - BCQ: Break-Point Conversion Quality (breaks_made / bp_chances)
  - TBP: Tiebreak Performance (tiebreaks_won / tiebreaks_played)
  - DSN: Deciding-Set Nerve (1 if won the deciding set, 0 if lost; NaN if none)

Outputs:
  - phi_match.csv
  - phi_weekly.csv
  - phi_weekly_surface.csv
"""

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TB_OR_WO = {"RET","W/O","WO","DEF","ABN"}
SET_PAT = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)")

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

def parse_sets_winner_pov(score: object):
    """
    Parse winner-oriented score string into list of (w_games, l_games) per set,
    skipping tokens like RET/W/O etc.
    """
    if not isinstance(score, str) or not score.strip():
        return []
    toks = [t for t in score.strip().split() if t.upper() not in TB_OR_WO]
    out = []
    for tok in toks:
        m = SET_PAT.match(tok)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2))))
    return out

def went_to_decider(score: object, best_of: object) -> int:
    if not isinstance(score, str) or pd.isna(best_of):
        return 0
    toks = [t for t in score.strip().split() if t.upper() not in TB_OR_WO]
    if not toks:
        return 0
    nsets = len(toks)
    max_sets = 5 if int(best_of) >= 5 else 3
    return int(nsets == max_sets)

def tiebreak_counts_winner_pov(score: object):
    """
    From a winner-oriented score string, count how many tiebreak sets (7-6 or 6-7)
    were won/lost by the match winner.
    """
    if not isinstance(score, str) or not score.strip():
        return (0, 0, 0)  # tb_played, tb_won_by_winner, tb_lost_by_winner
    toks = [t for t in score.strip().split() if t.upper() not in TB_OR_WO]
    tb_played = tb_w_winner = tb_l_winner = 0
    for tok in toks:
        if tok.startswith("7-6") or tok.startswith("6-7"):
            tb_played += 1
            if tok.startswith("7-6"):
                tb_w_winner += 1
            else:
                tb_l_winner += 1
    return (tb_played, tb_w_winner, tb_l_winner)

# ---------------------------
# Build long player–match table
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

    # Parse once on winner view
    sets_wl = m["score"].map(parse_sets_winner_pov)
    m["decider"] = [went_to_decider(s, b) for s, b in zip(m["score"], m["best_of"])]
    tb_trip = m["score"].map(tiebreak_counts_winner_pov)
    m["tb_played"] = [t[0] for t in tb_trip]
    m["tb_w_winner"] = [t[1] for t in tb_trip]
    m["tb_l_winner"] = [t[2] for t in tb_trip]

    # Winner perspective row
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
        "tourney_id":  m["tourney_id"],
        "round":       m["round"],
        "best_of":     m["best_of"],
        "score":       m["score"],
        "label":       1,
        # BP stats from player's serve / opponent's return
        "bp_faced":    m["w_bpFaced"],
        "bp_saved":    m["w_bpSaved"],
        "opp_bp_faced": m["l_bpFaced"],
        "opp_bp_saved": m["l_bpSaved"],
        # pressure contexts
        "decider":     m["decider"].astype(int),
        "tb_played":   m["tb_played"],
        "tb_won":      m["tb_w_winner"],   # winner of match won these TBs
        "tb_lost":     m["tb_l_winner"],
    })

    # Loser perspective row (invert TB wins/losses relative to match winner)
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
        "tourney_id":  m["tourney_id"],
        "round":       m["round"],
        "best_of":     m["best_of"],
        "score":       m["score"],
        "label":       0,
        "bp_faced":    m["l_bpFaced"],
        "bp_saved":    m["l_bpSaved"],
        "opp_bp_faced": m["w_bpFaced"],
        "opp_bp_saved": m["w_bpSaved"],
        "decider":     m["decider"].astype(int),
        "tb_played":   m["tb_played"],
        "tb_won":      m["tb_l_winner"],   # for loser row, TBs won = TBs the winner lost
        "tb_lost":     m["tb_w_winner"],
    })

    PM = pd.concat([W, L], ignore_index=True)
    PM = PM.sort_values(["player_id","date","tourney_id"]).reset_index(drop=True)

    # Break components
    PM["breaks_made"]   = (PM["opp_bp_faced"] - PM["opp_bp_saved"]).clip(lower=0)
    PM["bp_chances"]    = PM["opp_bp_faced"].clip(lower=0)
    PM["BSQ_raw"] = np.where(PM["bp_faced"] > 0, PM["bp_saved"] / PM["bp_faced"], np.nan)
    PM["BCQ_raw"] = np.where(PM["bp_chances"] > 0, PM["breaks_made"] / PM["bp_chances"], np.nan)
    # Tiebreak component
    PM["TBP_raw"] = np.where(PM["tb_played"] > 0, PM["tb_won"] / PM["tb_played"], np.nan)
    # Deciding-set nerve: for winners in decider -> 1, losers in decider -> 0, else NaN
    PM["DSN_raw"] = np.where(PM["decider"] == 1, PM["label"].astype(float), np.nan)

    # Clip to [0,1]
    for c in ["BSQ_raw","BCQ_raw","TBP_raw","DSN_raw"]:
        PM[c] = PM[c].clip(0, 1)

    return PM


# ---------------------------
# Baselines (prior 52 weeks)
# ---------------------------
def add_prior_52w_baselines(PM: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, sub in PM.groupby("player_id", sort=False):
        s = sub.sort_values("date").set_index("date").copy()

        for col in ["BSQ_raw","BCQ_raw","TBP_raw","DSN_raw"]:
            s[col + "_prior"] = s[col].shift(1)
            s[col + "_base"]  = s[col + "_prior"].rolling("365D", min_periods=8).mean()

        # conservative defaults
        s["BSQ_raw_base"] = s["BSQ_raw_base"].fillna(0.60)
        s["BCQ_raw_base"] = s["BCQ_raw_base"].fillna(0.35)
        s["TBP_raw_base"] = s["TBP_raw_base"].fillna(0.50)
        s["DSN_raw_base"] = s["DSN_raw_base"].fillna(0.50)

        out.append(s.reset_index())

    return pd.concat(out, ignore_index=True)


# ---------------------------
# Compute PHI per match
# ---------------------------
def compute_phi_match(PMb: pd.DataFrame,
                      w_b: float = 0.30,
                      w_c: float = 0.30,
                      w_t: float = 0.20,
                      w_d: float = 0.20) -> pd.DataFrame:
    df = PMb.copy()
    df["BSQ_norm"] = df["BSQ_raw"] - df["BSQ_raw_base"]
    df["BCQ_norm"] = df["BCQ_raw"] - df["BCQ_raw_base"]
    df["TBP_norm"] = df["TBP_raw"] - df["TBP_raw_base"]
    df["DSN_norm"] = df["DSN_raw"] - df["DSN_raw_base"]

    # dynamic reweight for missing components
    weights = np.vstack([
        (~df["BSQ_norm"].isna()).astype(float) * w_b,
        (~df["BCQ_norm"].isna()).astype(float) * w_c,
        (~df["TBP_norm"].isna()).astype(float) * w_t,
        (~df["DSN_norm"].isna()).astype(float) * w_d
    ]).T
    wsum = weights.sum(axis=1)
    wsum = np.where(wsum == 0, np.nan, wsum)

    comps = np.vstack([
        df["BSQ_norm"].fillna(0.0),
        df["BCQ_norm"].fillna(0.0),
        df["TBP_norm"].fillna(0.0),
        df["DSN_norm"].fillna(0.0)
    ]).T

    df["PHI_match"] = (weights * comps).sum(axis=1) / wsum
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
            PHI_mean=("PHI_match","mean"),
            BSQ_mean=("BSQ_norm","mean"),
            BCQ_mean=("BCQ_norm","mean"),
            TBP_mean=("TBP_norm","mean"),
            DSN_mean=("DSN_norm","mean"),
            tb_played=("tb_played","sum"),
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
    ap = argparse.ArgumentParser(description="Pressure Handling Index (PHI)")
    ap.add_argument("--master", required=True, help="Path to master parquet (e.g., D:\\Tennis\\data\\master\\tennis_master_1991.parquet)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--w_b", type=float, default=0.30, help="Weight for BSQ")
    ap.add_argument("--w_c", type=float, default=0.30, help="Weight for BCQ")
    ap.add_argument("--w_t", type=float, default=0.20, help="Weight for TBP")
    ap.add_argument("--w_d", type=float, default=0.20, help="Weight for DSN")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player–match table…")
    PM = build_long(master)

    print("[INFO] Adding 52-week baselines…")
    PMb = add_prior_52w_baselines(PM)

    print("[INFO] Computing match-level PHI…")
    PMc = compute_phi_match(PMb, args.w_b, args.w_c, args.w_t, args.w_d)

    # Match-level export
    match_cols = [
        "player_id","player_name","opp_id","opp_name",
        "date","surface","iso_year","week_num","week_start",
        "tourney_id","round","best_of","score","label",
        "bp_faced","bp_saved","opp_bp_faced","opp_bp_saved",
        "breaks_made","bp_chances",
        "tb_played","tb_won","tb_lost","decider",
        "BSQ_raw","BCQ_raw","TBP_raw","DSN_raw",
        "BSQ_raw_base","BCQ_raw_base","TBP_raw_base","DSN_raw_base",
        "BSQ_norm","BCQ_norm","TBP_norm","DSN_norm",
        "PHI_match"
    ]
    match_out = out_root / "phi_match.csv"
    PMc[match_cols].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level PHI → {match_out} ({len(PMc):,} rows)")

    # Weekly global
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(PMc, by_surface=False)
    weekly_out = out_root / "phi_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly PHI → {weekly_out} ({len(weekly):,} rows)")

    # Weekly by surface
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(PMc, by_surface=True)
    weekly_surf_out = out_root / "phi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface PHI → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
