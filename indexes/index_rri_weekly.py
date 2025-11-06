"""
Recovery Rate Index (RRI) — Weekly Series (CSV)
===============================================

Quantifies how well a player maintains/regains performance under fatigue
(long matches and/or short rest).

Definitions
-----------
Fatigue Index (FI) = 0.5 * minutes_norm + 0.5 * (matches_last3d / 3), clipped to [0,1]
  - minutes_norm = minutes / 180 (capped to 1.0)
  - matches_last3d = number of prior matches within the last 3 days

Fatigue levels:
  rested   : FI in [0.00, 0.33]
  moderate : FI in (0.33, 0.66]
  fatigued : FI in (0.66, 1.00]

RRI (player summary):
  RRI_raw      = win_rate(fatigued) / (win_rate(rested) + 1e-6)
  RRI_weighted = RRI_raw * mean(tier_weight)

Weekly series:
  For each player-week:
    RRI_weekly = win_rate(fatigued that week) / (win_rate(rested that week) + 1e-6)
    mean_fatigue_index = average FI for that player's matches that week

Outputs
-------
- rri_events.csv   : one row per player (overall summary across all matches)
- rri_weekly.csv   : weekly time series per player
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Config / constants
# -----------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
FATIGUE_BINS = [-0.01, 0.33, 0.66, 1.00]
FATIGUE_LABELS = ["rested", "moderate", "fatigued"]


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
    """Week bucket starting Monday (Timestamp of week start)."""
    return dt.dt.to_period("W-MON").dt.start_time


def _long_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winner/loser -> player rows, robust date parsing, 1991+ filter,
    and tier normalization. Keeps only columns needed here.
    """
    df = df.copy()

    # robust date
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # tier
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)
    df["tier_weight"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    # minutes (cap to 600 raw; then normalize to /180 clip 0..1 later)
    minutes = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0).clip(lower=0, upper=600)

    # base columns for both winner and loser sides
    base = {
        "date": df["tourney_date"],
        "week": _to_week(df["tourney_date"]),
        "tier_weight": df["tier_weight"],
        "minutes": minutes,
    }

    # winner rows
    W = pd.DataFrame({
        **base,
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "label": 1,
    })
    # loser rows
    L = pd.DataFrame({
        **base,
        "player_id": df["loser_id"],
        "player_name": df["loser_name"],
        "label": 0,
    })

    long = pd.concat([W, L], ignore_index=True)
    long = long.dropna(subset=["player_id", "date"])
    # ensure types
    long["player_id"] = long["player_id"].astype(str)
    return long.sort_values(["player_id", "date"]).reset_index(drop=True)


def _compute_matches_last3d_fast(dates: np.ndarray) -> np.ndarray:
    """
    For a sorted array of datetimes (for one player), compute for each index i
    the number of *previous* matches within the last 3 days.
    Uses binary search (searchsorted) → O(n log n).
    """
    # Convert to int64 nanoseconds for speed
    t = dates.astype("datetime64[ns]").astype("int64")
    three_days_ns = np.int64(3 * 24 * 3600 * 1e9)

    out = np.zeros(len(t), dtype=np.int16)
    for i in range(len(t)):
        thresh = t[i] - three_days_ns
        j = np.searchsorted(t, thresh, side="right")
        out[i] = i - j  # number of prior matches strictly within 3 days window
    return out


def _enrich_with_fatigue(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Add minutes_norm, matches_last3d, fatigue_index, fatigue_level to df_long.
    Done per player in a vectorized way.
    """
    df = df_long.copy()

    # minutes_norm per player (avoid div by 0)
    df["minutes_norm"] = 0.0
    for pid, g in df.groupby("player_id", sort=False):
        m = g["minutes"].to_numpy(dtype=float)
        denom = max(180.0, np.nanmax(m))  # at least 180 for scaling
        df.loc[g.index, "minutes_norm"] = np.clip(m / 180.0, 0.0, 1.0)

    # matches_last3d per player
    df["matches_last3d"] = 0
    for pid, g in df.groupby("player_id", sort=False):
        idx = g.index
        arr = g["date"].to_numpy()
        last3 = _compute_matches_last3d_fast(arr)
        df.loc[idx, "matches_last3d"] = last3

    # fatigue_index
    df["fatigue_index"] = 0.5 * df["minutes_norm"] + 0.5 * np.clip(df["matches_last3d"] / 3.0, 0.0, 1.0)
    df["fatigue_index"] = df["fatigue_index"].clip(0.0, 1.0)

    # fatigue_level
    df["fatigue_level"] = pd.cut(df["fatigue_index"], bins=FATIGUE_BINS, labels=FATIGUE_LABELS, include_lowest=True)

    return df


def _player_summary_rri(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build one summary row per player:
      rested_wr, fatigued_wr, RRI_raw, RRI_weighted, matches, mean_fatigue_index
    """
    # Suppress future warning by passing observed=True explicitly
    grp = enriched.groupby(["player_id", "player_name", "fatigue_level"], observed=True)["label"].agg(["mean", "count"]).reset_index()
    # pivot to get rested/fatigued columns
    pvt = grp.pivot(index=["player_id", "player_name"], columns="fatigue_level", values="mean").reset_index()
    # ensure columns exist
    for c in FATIGUE_LABELS:
        if c not in pvt.columns:
            pvt[c] = np.nan

    pvt = pvt.rename(columns={"rested": "rested_wr", "fatigued": "fatigued_wr"})
    pvt["rested_wr"] = pvt["rested_wr"].fillna(0.0)
    pvt["fatigued_wr"] = pvt["fatigued_wr"].fillna(pvt["rested_wr"])

    # mean tier weight and mean fatigue index per player
    agg_more = enriched.groupby("player_id", observed=True).agg(
        tier_mean=("tier_weight", "mean"),
        matches=("label", "size"),
        mean_fatigue_index=("fatigue_index", "mean"),
    ).reset_index()

    out = pvt.merge(agg_more, on="player_id", how="left")
    out["RRI_raw"] = out["fatigued_wr"] / (out["rested_wr"] + 1e-6)
    out["RRI_weighted"] = out["RRI_raw"] * out["tier_mean"]

    cols = ["player_id", "player_name", "rested_wr", "fatigued_wr", "RRI_raw", "RRI_weighted", "matches", "mean_fatigue_index"]
    return out[cols]


def _weekly_series(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly RRI per player:
      RRI_weekly = WR_fatigued_week / (WR_rested_week + 1e-6)
      plus matches and mean_fatigue_index per week.
    """
    # compute weekly WR by fatigue level
    grouped = (
        enriched
        .groupby(["player_id", "player_name", "week", "fatigue_level"], observed=True)
        .agg(win_rate=("label", "mean"), matches=("label", "count"), mean_fatigue_index=("fatigue_index", "mean"))
        .reset_index()
    )

    rested = grouped[grouped["fatigue_level"] == "rested"][["player_id", "player_name", "week", "win_rate"]]
    rested = rested.rename(columns={"win_rate": "wr_rested"})

    fatig = grouped[grouped["fatigue_level"] == "fatigued"][["player_id", "player_name", "week", "win_rate"]]
    fatig = fatig.rename(columns={"win_rate": "wr_fatigued"})

    # combine, allow weeks where one side missing (drop rows without both WRs)
    merged = rested.merge(fatig, on=["player_id", "player_name", "week"], how="inner")

    # matches and mean FI per week (regardless of level)
    weekly_load = (
        enriched.groupby(["player_id", "player_name", "week"], observed=True)
        .agg(matches=("label", "count"), mean_fatigue_index=("fatigue_index", "mean"))
        .reset_index()
    )

    out = merged.merge(weekly_load, on=["player_id", "player_name", "week"], how="left")
    out["RRI_weekly"] = out["wr_fatigued"] / (out["wr_rested"] + 1e-6)

    # order
    out = out.sort_values(["player_id", "week"]).reset_index(drop=True)
    return out[["player_id", "player_name", "week", "RRI_weekly", "matches", "mean_fatigue_index"]]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute Recovery Rate Index (RRI) weekly CSVs")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-level long table...")
    df_long = _long_from_master(master)
    print(f"[INFO] Rows: {len(df_long):,} | Players: {df_long['player_id'].nunique():,}")

    print("[INFO] Computing fatigue metrics...")
    enriched = _enrich_with_fatigue(df_long)

    # ---- Player summary (events) ----
    print("[INFO] Computing player-level RRI summary...")
    rri_events = _player_summary_rri(enriched)
    events_path = Path(args.out_root) / "rri_events.csv"
    rri_events.to_csv(events_path, index=False)
    print(f"[INFO] Saved player-level RRI summary → {events_path} ({len(rri_events):,} rows)")

    # ---- Weekly series ----
    print("[INFO] Computing weekly RRI series...")
    weekly = _weekly_series(enriched)
    weekly_path = Path(args.out_root) / "rri_weekly.csv"
    # Always write a CSV, even if empty
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly RRI series → {weekly_path} ({len(weekly):,} rows)")

if __name__ == "__main__":
    main()
