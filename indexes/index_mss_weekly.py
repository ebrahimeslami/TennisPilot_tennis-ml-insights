"""
Momentum Sustainability Score (MSS) — Weekly Series
===================================================

Measures how long players sustain high form after a strong performance
(momentum event). A momentum event is defined as when a player's recent
rolling win rate exceeds their baseline by at least one standard deviation.

MSS quantifies the fraction of subsequent weeks where the player's win rate
remains above their baseline — capturing how durable their momentum is.

Formula
-------
    MSS = Weeks_above_baseline_after_event / Total_weeks_tracked

Optionally:
    MSS_weighted = MSS × TournamentWeight × OpponentStrengthFactor

Interpretation
--------------
> 0.7 → Sustains form for long periods (consistent elite)
0.4–0.7 → Moderate sustainability
< 0.4 → Form fades quickly after peaks (streaky player)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# Tournament level weights (optional)
# ---------------------------------------------------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
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

def _to_week(dt):
    return dt.dt.to_period("W-MON").dt.start_time

def _long_from_master(df):
    """Winner/loser → player-level with weekly field."""
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)

    # Winner rows
    W = pd.DataFrame({
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "date": df["tourney_date"],
        "week": _to_week(df["tourney_date"]),
        "tier": df["tier_norm"],
        "label": 1,
        "tourney_id": df["tourney_id"],
    })
    # Loser rows
    L = W.copy()
    L["player_id"], L["player_name"], L["label"] = df["loser_id"], df["loser_name"], 0
    long = pd.concat([W, L], ignore_index=True).dropna(subset=["player_id", "date"])
    return long.sort_values(["player_id", "date"])

# ---------------------------------------------------------------------
# Core MSS computation
# ---------------------------------------------------------------------
def _compute_mss_for_player(df_p, window=5, n_forward=8):
    """
    Compute MSS for a single player.
    - Rolling baseline of last `window` tournaments.
    - Identify momentum events.
    - Track next `n_forward` tournaments for sustainability.
    """
    df_p = df_p.sort_values("date").reset_index(drop=True)

    # Rolling win rate baseline
    df_p["rolling_wr"] = df_p["label"].rolling(window=window, min_periods=3).mean()
    df_p["baseline_mean"] = df_p["rolling_wr"].expanding().mean()
    df_p["baseline_std"] = df_p["rolling_wr"].expanding().std()
    df_p["momentum_event"] = df_p["rolling_wr"] > (df_p["baseline_mean"] + df_p["baseline_std"])

    events = []
    for i, row in df_p[df_p["momentum_event"]].iterrows():
        start_date = row["date"]
        after = df_p[(df_p["date"] > start_date)].head(n_forward)
        if after.empty:
            continue
        above_baseline = (after["rolling_wr"] > after["baseline_mean"]).sum()
        sustain_ratio = above_baseline / len(after)

        tier_mode = after["tier"].mode().iloc[0] if not after["tier"].mode().empty else "B"
        tier_w = TIER_WEIGHT.get(tier_mode, 1.0)
        mss_weighted = sustain_ratio * tier_w

        events.append({
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "event_date": start_date,
            "event_week": row["week"],
            "momentum_duration_weeks": len(after),
            "weeks_above_baseline": int(above_baseline),
            "MSS": sustain_ratio,
            "MSS_weighted": mss_weighted,
        })
    return pd.DataFrame(events)

def _expand_to_weekly(df_events, all_weeks, halflife_weeks=12):
    """Create continuous weekly MSS via EWMA smoothing."""
    out = []
    for pid, dfe in df_events.groupby("player_id"):
        weeks = pd.DataFrame({"week": all_weeks})
        weeks["player_id"] = pid
        d = weeks.merge(
            dfe[["player_id", "event_week", "MSS_weighted"]],
            left_on=["player_id", "week"],
            right_on=["player_id", "event_week"],
            how="left"
        )
        d["signal"] = d["MSS_weighted"].fillna(0.0)
        d["mss_weekly_smoothed"] = d["signal"].ewm(halflife=halflife_weeks, adjust=False).mean()
        d = d[["player_id", "week", "mss_weekly_smoothed"]]
        out.append(d)
    return pd.concat(out, ignore_index=True)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute Momentum Sustainability Score (MSS)")
    parser.add_argument("--master", required=True, help="Path to master parquet")
    parser.add_argument("--out_root", required=True, help="Output directory")
    parser.add_argument("--window", type=int, default=5, help="Rolling window for win rate baseline")
    parser.add_argument("--n_forward", type=int, default=8, help="Number of tournaments forward to check sustainability")
    parser.add_argument("--ewm_halflife_weeks", type=int, default=12, help="EWMA smoothing half-life (weeks)")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.master} ...")
    master = pd.read_parquet(args.master)
    df_long = _long_from_master(master)
    print(f"[INFO] Matches: {len(df_long):,} | Players: {df_long['player_id'].nunique():,}")

    # Compute MSS event data
    all_events = []
    for pid, dfp in tqdm(df_long.groupby("player_id"), desc="Computing MSS"):
        ev = _compute_mss_for_player(dfp, window=args.window, n_forward=args.n_forward)
        if not ev.empty:
            all_events.append(ev)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    # Save event-level results
    events_path = out_root / "mss_events.csv"
    events.to_csv(events_path, index=False)
    print(f"[INFO] Saved MSS events → {events_path} ({len(events):,} rows)")

    # Continuous weekly MSS (smoothed)
    if not events.empty:
        all_weeks = _to_week(df_long["date"]).dropna().sort_values().unique()
        all_weeks = pd.Series(pd.to_datetime(all_weeks))
        weekly = _expand_to_weekly(events, all_weeks, halflife_weeks=args.ewm_halflife_weeks)
        weekly_path = out_root / "mss_weekly.csv"
        weekly.to_csv(weekly_path, index=False)
        print(f"[INFO] Saved weekly MSS → {weekly_path} ({len(weekly):,} rows)")
    else:
        print("[WARN] No momentum events detected → no weekly MSS file created.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
