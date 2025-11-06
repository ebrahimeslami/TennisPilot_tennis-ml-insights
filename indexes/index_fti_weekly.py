# -*- coding: utf-8 -*-
"""
Form Trajectory Index (FTI)
---------------------------
Purpose:
    Quantifies a player's *current trajectory* by comparing short-term
    performance to a longer-term baseline, with opponent-quality adjustment,
    recency emphasis (EWMA), and recent match-importance weighting.

Outputs:
    - fti_match.csv          (player–match level, 1 row per player per match)
    - fti_weekly.csv         (player–week aggregate across surfaces)
    - fti_weekly_surface.csv (player–week per surface)

CLI:
    python index_fti_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" \
        --out_root "D:\\Tennis\\data\\indexes" --short_w 8 --long_w 52 --alpha 0.5

Notes:
    - Works on Jeff Sackmann-style ATP/WTA master files combined.
    - Requires columns typical in your master: tourney_date, surface, tourney_level,
      winner_id/winner_name/winner_rank, loser_id/loser_name/loser_rank.
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    """Robustly parse yyyymmdd held as int/str/object → datetime64[ns]."""
    s = series.copy()
    # Coerce to string of 8 digits when possible
    def _cast(x):
        if pd.isna(x):
            return np.nan
        try:
            xi = int(x)
            return f"{xi:08d}"
        except Exception:
            xs = str(x)
            return xs if len(xs) == 8 and xs.isdigit() else np.nan

    s = s.map(_cast)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def canon_surface(s):
    if not isinstance(s, str):
        return "Other"
    st = s.strip().lower()
    if "hard" in st:
        return "Hard"
    if "clay" in st:
        return "Clay"
    if "grass" in st:
        return "Grass"
    if "carpet" in st or "indoor" in st:
        return "Carpet"
    return "Other"


def level_weight(x: str) -> float:
    """
    Map tourney_level codes to importance in [0,1].
    Adjust if your master uses different codes; this is robust and conservative.
    """
    if not isinstance(x, str):
        return 0.4
    xl = x.strip().upper()
    # Common codes: G (GS), M (Masters 1000), A (ATP 500), B (ATP 250),
    # D/C (Davis/Chall), L (Laver/United), etc.
    if xl.startswith("G"):
        return 1.00
    if xl.startswith("M"):
        return 0.85
    if xl.startswith("A") or "500" in xl:
        return 0.70
    if xl.startswith("B") or "250" in xl:
        return 0.55
    if xl in {"L", "T", "U"}:
        return 0.50
    if xl.startswith("C") or "CH" in xl:
        return 0.45
    return 0.40


def build_long(master: pd.DataFrame) -> pd.DataFrame:
    """Winner+Loser → long format with dates, surface, opp ranks, level."""
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["level_w"] = m["tourney_level"].map(level_weight)

    W = pd.DataFrame({
        "player_id": m["winner_id"].astype(str),
        "player_name": m["winner_name"],
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "level_wt": m["level_w"],
        "label": 1,
        "opp_rank": m["loser_rank"],
    })
    L = pd.DataFrame({
        "player_id": m["loser_id"].astype(str),
        "player_name": m["loser_name"],
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "level_wt": m["level_w"],
        "label": 0,
        "opp_rank": m["winner_rank"],
    })
    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["player_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)

    # Week keys
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    return df


def opponent_weight(opp_rank):
    """Map opponent rank to [0.5,1.0] (tougher opponent ⇒ higher weight)."""
    try:
        r = float(opp_rank)
    except Exception:
        r = np.nan
    if np.isnan(r):
        r = 200.0
    r = max(1.0, min(200.0, r))
    return 0.5 + 0.5 * (200.0 - r) / 200.0


def compute_fti(df_long: pd.DataFrame, short_w: int = 8, long_w: int = 52, alpha: float = 0.5, min_matches: int = 3) -> pd.DataFrame:
    """Compute match-level FTI components and FTI for each player row."""
    df = df_long.copy()
    # Opponent-adjusted outcome
    df["opp_w"] = df["opp_rank"].map(opponent_weight)
    df["y_adj"] = df["label"] * df["opp_w"]

    def per_player(g):
        g = g.sort_values("date").copy()
        # Rolling means
        g["SWR"] = g["y_adj"].rolling(short_w, min_periods=min_matches).mean()
        g["LWR"] = g["y_adj"].rolling(long_w, min_periods=min_matches).mean()
        # Momentum gap (-1..1), later rescaled to 0..1
        g["gap"] = (g["SWR"] - g["LWR"]).clip(-1, 1)

        # Recency emphasis (EWMA of *raw* outcomes)
        g["EW"] = g["label"].ewm(alpha=alpha, adjust=False).mean()

        # Recent importance (short window)
        g["IMP"] = g["level_wt"].rolling(short_w, min_periods=min_matches).mean().fillna(0.5)

        # Rescale gap to [0,1]
        g["gap01"] = (g["gap"] + 1.0) / 2.0

        # Composite FTI
        g["FTI"] = (0.5 * g["gap01"] + 0.3 * g["EW"] + 0.2 * g["IMP"]).clip(0, 1)

        return g

    out = df.groupby("player_id", group_keys=False).apply(per_player)

    # Final tidy columns
    cols = [
        "player_id", "player_name", "date", "iso_year", "week_num", "week_start",
        "surface", "label", "opp_rank",
        "SWR", "LWR", "gap", "EW", "IMP", "FTI"
    ]
    return out[cols].reset_index(drop=True)


def weekly_aggregate(df: pd.DataFrame, by_surface: bool = False) -> pd.DataFrame:
    """Aggregate FTI by player-week (optionally per surface)."""
    # Ensure week_start exists
    if "week_start" not in df.columns:
        if {"iso_year", "week_num"}.issubset(df.columns):
            df["week_start"] = df.apply(
                lambda r: datetime.fromisocalendar(int(r["iso_year"]), int(r["week_num"]), 1),
                axis=1,
            )
        else:
            raise KeyError("Missing iso_year/week_num to build week_start.")

    keys = ["player_id", "iso_year", "week_num"]
    if by_surface:
        keys.append("surface")

    agg = (
        df.groupby(keys, observed=True)
          .agg(
              player_name=("player_name", "last"),
              week_start=("week_start", "first"),
              matches=("label", "count"),
              win_rate=("label", "mean"),
              SWR=("SWR", "mean"),
              LWR=("LWR", "mean"),
              gap=("gap", "mean"),
              EW=("EW", "mean"),
              IMP=("IMP", "mean"),
              FTI=("FTI", "mean"),
          )
          .reset_index()
          .sort_values(keys)
          .reset_index(drop=True)
    )
    return agg


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Form Trajectory Index (FTI) — match/weekly/weekly_surface CSVs")
    ap.add_argument("--master", required=True, help="Path to master parquet")
    ap.add_argument("--out_root", required=True, help="Output folder for CSVs")
    ap.add_argument("--short_w", type=int, default=8, help="Short window (matches)")
    ap.add_argument("--long_w", type=int, default=52, help="Long window (matches)")
    ap.add_argument("--alpha", type=float, default=0.5, help="EWMA alpha for recency")
    ap.add_argument("--min_matches", type=int, default=3, help="Min matches to compute rolling means")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building long-format table…")
    df_long = build_long(master)

    print("[INFO] Computing FTI components and composite…")
    df_match = compute_fti(
        df_long,
        short_w=args.short_w,
        long_w=args.long_w,
        alpha=args.alpha,
        min_matches=args.min_matches,
    )

    # Save match-level
    match_path = out_root / "fti_match.csv"
    df_match.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level FTI → {match_path} ({len(df_match):,} rows)")

    # Weekly (global)
    print("[INFO] Aggregating weekly (global)…")
    weekly = weekly_aggregate(df_match, by_surface=False)
    weekly_path = out_root / "fti_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly FTI → {weekly_path} ({len(weekly):,} rows)")

    # Weekly (surface)
    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = weekly_aggregate(df_match, by_surface=True)
    weekly_surf_path = out_root / "fti_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface FTI → {weekly_surf_path} ({len(weekly_surf):,} rows)")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
