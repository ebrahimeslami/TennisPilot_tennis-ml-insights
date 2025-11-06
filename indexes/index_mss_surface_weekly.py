"""
Momentum Sustainability Score (MSS) — Surface-Specific Weekly (1991–2025)
==========================================================================

Extension of MSS index with per-surface aggregation.
Keeps the same core formulation but computes baselines, sustainability,
and streak continuity *independently per surface type*.

Outputs
-------
- mss_weekly_surface.csv : player_id, player_name, surface, iso_year, week_num, week_start,
                           matches, week_wr, past_mean_wr, past_std_wr,
                           sustain, z_sustain, streak_prev, week_tier_mean, tier_delta, MSS

Run
---
python index_mss_surface_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" ^
                                   --out_root "D:\\Tennis\\data\\indexes" ^
                                   --weeks_window 8
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


# ---------------- Build player-match ----------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    tier_raw = df.get("tourney_level_norm", df.get("tourney_level", ""))
    tier_w = _tier_weight(pd.Series(tier_raw, index=df.index))
    date = df["tourney_date"]
    iso_year = _iso_year(date)
    weekn = _week_num(date)
    week_start = _week_start(date)
    surface = df["surface"].fillna("Hard").str.capitalize()

    # Winner perspective
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "label": 1,
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "surface": surface,
        "tier_w": tier_w,
    })

    # Loser perspective
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "label": 0,
        "date": date,
        "iso_year": iso_year,
        "week_num": weekn,
        "week_start": week_start,
        "surface": surface,
        "tier_w": tier_w,
    })

    pm = pd.concat([W, L], ignore_index=True)
    pm = pm.dropna(subset=["player_id", "opp_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)
    return pm


# ---------------- Weekly aggregation (with surface) ----------------
def aggregate_weekly(pm: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby(["player_id","player_name","surface","iso_year","week_num","week_start"], observed=True)
    wk = g.agg(
        matches=("label", "size"),
        week_weighted_wins=("label", lambda s: float((pm.loc[s.index, "tier_w"].values * s.values).sum())),
        week_weighted_total=("tier_w", "sum"),
        week_tier_mean=("tier_w", "mean"),
    ).reset_index()

    wk["week_wr"] = np.where(wk["week_weighted_total"] > 0, wk["week_weighted_wins"] / wk["week_weighted_total"], 0.0)

    # player-surface-level career mean tier weight
    career_tier_mean = (
        pm.groupby(["player_id","surface"], as_index=False)["tier_w"].mean()
          .rename(columns={"tier_w": "player_mean_tier_w"})
    )
    wk = wk.merge(career_tier_mean, on=["player_id","surface"], how="left")
    wk["player_mean_tier_w"] = wk["player_mean_tier_w"].fillna(1.0)
    wk["tier_delta"] = wk["week_tier_mean"] - wk["player_mean_tier_w"]

    # Rolling MSS per player+surface
    def _roll(g):
        g = g.sort_values(["iso_year","week_num"]).reset_index(drop=True)
        g["past_mean_wr"] = g["week_wr"].rolling(window=8, min_periods=1).mean().shift(1)
        g["past_std_wr"]  = g["week_wr"].rolling(window=8, min_periods=1).std(ddof=0).shift(1)
        g["past_mean_wr"] = g["past_mean_wr"].fillna(0.5)
        g["past_std_wr"]  = g["past_std_wr"].fillna(0.1)
        g["sustain"]   = g["week_wr"] - g["past_mean_wr"]
        g["z_sustain"] = g["sustain"] / np.maximum(g["past_std_wr"], 0.1)
        g["MSS"] = 0.75*g["z_sustain"] + 0.25*g["tier_delta"]
        return g

    out = wk.groupby(["player_id","player_name","surface"], group_keys=False).apply(_roll).reset_index(drop=True)
    return out


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Momentum Sustainability Score (MSS) — Surface Weekly CSV")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--weeks_window", type=int, default=8)
    ap.add_argument("--write_match_csv", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match table...")
    pm = build_player_match(master)
    if args.write_match_csv:
        pm.to_csv(out_root / "mss_match_surface.csv", index=False)
        print(f"[INFO] Saved match-level file → {out_root / 'mss_match_surface.csv'}")

    print(f"[INFO] Matches: {len(pm):,} | Players: {pm['player_id'].nunique()}")

    print("[INFO] Aggregating weekly MSS by surface...")
    mss_surface = aggregate_weekly(pm)

    out_path = out_root / "mss_weekly_surface.csv"
    mss_surface.to_csv(out_path, index=False)
    print(f"[INFO] Saved MSS per surface → {out_path} ({len(mss_surface):,} rows)")

    if not mss_surface.empty:
        years = mss_surface["iso_year"]
        print(f"[INFO] Coverage: {int(years.min())}-{int(years.max())} | Players: {mss_surface['player_id'].nunique()}")


if __name__ == "__main__":
    main()
