"""
Surface Transition Index (STI) — Weekly Series (CSV Version)
============================================================

Computes STI when a player switches surfaces and produces:
  1) Event-level weekly STI at the transition week (sti_events_weekly.csv)
  2) Continuous weekly STI series per player (sti_weekly_smoothed.csv)

Definitions
-----------
WR_before: mean(win) over the last N tournaments on the previous surface before the switch
WR_after : mean(win) over the first N tournaments on the new surface after the switch

STI_raw = 1 - |WR_after - WR_before| / (WR_before + 1e-6)
STI_weighted = STI_raw × TierWeight × SurfacePairWeight
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Weight dictionaries
# ---------------------------------------------------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
SURFACE_WEIGHT = {
    ("Clay", "Grass"): 1.3,
    ("Grass", "Hard"): 1.2,
    ("Hard", "Clay"): 1.4,
    ("Clay", "Hard"): 1.1,
    ("Grass", "Clay"): 1.5,
}

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def _normalize_tourney_level(x: str) -> str:
    """Map raw tournament levels to G, M, A, or B."""
    if not isinstance(x, str):
        return "B"
    x = x.strip().upper()
    if "G" in x or "SLAM" in x:
        return "G"
    if "M" in x or "1000" in x or "MAST" in x:
        return "M"
    if "A" in x or "500" in x:
        return "A"
    return "B"

def _to_week(series: pd.Series) -> pd.Series:
    """Convert datetime to week start (Monday)."""
    return series.dt.to_period("W-MON").dt.start_time

def _long_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """Winner/loser -> player rows; robust date parsing."""
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # Normalize level
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)

    # Winner rows
    W = pd.DataFrame({
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "date": df["tourney_date"],
        "week": _to_week(df["tourney_date"]),
        "surface": df["surface"],
        "tier": df["tier_norm"],
        "label": 1,
        "tourney_id": df["tourney_id"],
    })

    # Loser rows
    L = W.copy()
    L["player_id"], L["player_name"], L["label"] = df["loser_id"], df["loser_name"], 0

    df_long = pd.concat([W, L], ignore_index=True)
    df_long = df_long.dropna(subset=["player_id", "surface", "date"])
    df_long = df_long.sort_values(["player_id", "date"])
    return df_long

def _event_sti_for_player(dfp: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute STI for one player's surface transitions."""
    dfp = dfp.sort_values("date").reset_index(drop=True)
    dfp["prev_surface"] = dfp["surface"].shift(1)
    dfp["surface_change"] = dfp["surface"] != dfp["prev_surface"]

    out = []
    for i, row in dfp[dfp["surface_change"]].iterrows():
        if pd.isna(row["prev_surface"]):
            continue
        prev_surf, next_surf = str(row["prev_surface"]), str(row["surface"])
        date_change = row["date"]

        before = dfp[(dfp["surface"] == prev_surf) & (dfp["date"] < date_change)].tail(window)
        after  = dfp[(dfp["surface"] == next_surf) & (dfp["date"] > date_change)].head(window)
        if len(before) == 0 or len(after) == 0:
            continue

        wr_before = before["label"].mean()
        wr_after  = after["label"].mean()
        sti_raw   = 1.0 - abs(wr_after - wr_before) / (wr_before + 1e-6)

        tier_mode = after["tier"].mode().iloc[0] if not after["tier"].mode().empty else "B"
        tier_w    = TIER_WEIGHT.get(tier_mode, 1.0)
        surf_w    = SURFACE_WEIGHT.get((prev_surf, next_surf), 1.0)
        sti_w     = sti_raw * tier_w * surf_w

        out.append({
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "event_date": date_change,
            "event_week": row["week"],
            "prev_surface": prev_surf,
            "next_surface": next_surf,
            "wr_before": wr_before,
            "wr_after": wr_after,
            "STI_raw": float(sti_raw),
            "STI_weighted": float(sti_w),
            "tier_mode": tier_mode,
        })
    return pd.DataFrame(out)

def _expand_to_weekly(df_events: pd.DataFrame, all_weeks: pd.Series, halflife_weeks: int = 12) -> pd.DataFrame:
    """Build continuous weekly series using EWMA of recent transitions."""
    out_frames = []
    for pid, dfe in df_events.groupby("player_id"):
        weeks = pd.DataFrame({"week": all_weeks})
        weeks["player_id"] = pid

        d = weeks.merge(
            dfe[["player_id", "event_week", "STI_weighted"]],
            left_on=["player_id", "week"],
            right_on=["player_id", "event_week"],
            how="left",
        )
        d["signal"] = d["STI_weighted"].fillna(0.0)
        d["sti_weekly_smoothed"] = d["signal"].ewm(halflife=halflife_weeks, adjust=False).mean()
        d = d[["player_id", "week", "sti_weekly_smoothed"]]
        out_frames.append(d)

    return pd.concat(out_frames, ignore_index=True)

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute weekly Surface Transition Index (STI)")
    parser.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    parser.add_argument("--out_root", required=True, help="Output folder")
    parser.add_argument("--window", type=int, default=3, help="N tournaments before/after switch")
    parser.add_argument("--ewm_halflife_weeks", type=int, default=12, help="EWMA halflife in weeks")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.master} ...")
    master = pd.read_parquet(args.master)
    df_long = _long_from_master(master)
    print(f"[INFO] Matches: {len(df_long):,} | Players: {df_long['player_id'].nunique():,}")

    # --- Event-level STI ---
    events = []
    for pid, dfp in df_long.groupby("player_id"):
        ev = _event_sti_for_player(dfp, window=args.window)
        if not ev.empty:
            events.append(ev)
    events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()

    events_path = out_root / "sti_events_weekly.csv"
    events.to_csv(events_path, index=False)
    print(f"[INFO] Saved transition events → {events_path} ({len(events):,} rows)")

    # --- Continuous weekly STI (EWMA) ---
    if not events.empty:
        all_weeks = _to_week(df_long["date"]).dropna().sort_values().unique()
        all_weeks = pd.Series(pd.to_datetime(all_weeks))
        weekly = _expand_to_weekly(events, all_weeks, halflife_weeks=args.ewm_halflife_weeks)

        weekly_path = out_root / "sti_weekly_smoothed.csv"
        weekly.to_csv(weekly_path, index=False)
        print(f"[INFO] Saved smoothed weekly STI → {weekly_path} ({len(weekly):,} rows)")
    else:
        print("[WARN] No surface transitions detected → no weekly output.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
