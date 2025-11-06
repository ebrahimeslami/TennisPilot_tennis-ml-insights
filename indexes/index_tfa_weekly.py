# -*- coding: utf-8 -*-
"""
Transition Fatigue Adaptation (TFA)
-----------------------------------
Combines physical (travel) and environmental (surface) transition effects
to measure how well players maintain performance after transitions.

Derived from:
 - TFC: Travel Fatigue Coefficient (distance/time zone)
 - STI: Surface Transition Index (surface adaptation)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# Helper functions
# -----------------------------
def to_datetime_yyyymmdd(series):
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str):
        return "Other"
    s = s.strip().lower()
    if "hard" in s:
        return "Hard"
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    if "carpet" in s or "indoor" in s:
        return "Carpet"
    return "Other"

SURFACE_DELTA = {
    ("Hard", "Clay"): 0.7, ("Hard", "Grass"): 0.5, ("Hard", "Carpet"): 0.3,
    ("Clay", "Hard"): 0.7, ("Clay", "Grass"): 0.8, ("Clay", "Carpet"): 0.4,
    ("Grass", "Hard"): 0.5, ("Grass", "Clay"): 0.8, ("Grass", "Carpet"): 0.6,
    ("Carpet", "Hard"): 0.3, ("Carpet", "Clay"): 0.4, ("Carpet", "Grass"): 0.6
}

def surface_delta(s1, s2):
    if s1 == s2:
        return 0.0
    return SURFACE_DELTA.get((s1, s2), 0.5)

# -----------------------------
# Core computation
# -----------------------------
def compute_tfa(master, tfc_path, sti_path):
    df_tfc = pd.read_csv(tfc_path)
    df_sti = pd.read_csv(sti_path)

    # Standardize ID types
    for d in [df_tfc, df_sti]:
        d["player_id"] = d["player_id"].astype(str)

    # Identify TFC and STI column names dynamically
    tfc_col = "TFC" if "TFC" in df_tfc.columns else [c for c in df_tfc.columns if "TFC" in c][0]
    sti_col = [c for c in df_sti.columns if "STI" in c][0] if any("STI" in c for c in df_sti.columns) else None
    if sti_col is None:
        raise ValueError(f"No 'STI' column found in {sti_path}. Available columns: {df_sti.columns.tolist()}")

    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df["surface_c"] = df["surface"].map(canon_surface)
    df["iso_year"] = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Expand to player level
    players = []
    for label in ["winner", "loser"]:
        tmp = pd.DataFrame({
            "player_id": df[f"{label}_id"].astype(str),
            "player_name": df[f"{label}_name"],
            "date": df["tourney_date"],
            "surface": df["surface_c"],
            "label": 1 if label == "winner" else 0,
            "opp_rank": df["loser_rank"] if label == "winner" else df["winner_rank"],
        })
        players.append(tmp)

    df_long = pd.concat(players, ignore_index=True)

    # Merge TFC (physical fatigue) and STI (surface adaptation)
    # --- Harmonize TFC columns ---
    # Ensure iso_year exists in TFC data
    if "iso_year" not in df_tfc.columns:
        if "week_start" in df_tfc.columns:
            df_tfc["iso_year"] = pd.to_datetime(df_tfc["week_start"], errors="coerce").dt.isocalendar().year.astype(
                "Int64")
        elif "year" in df_tfc.columns:
            df_tfc["iso_year"] = df_tfc["year"].astype("Int64")
        else:
            raise KeyError(f"No 'iso_year' or equivalent date column found in {tfc_path}")

    # Ensure week_num exists
    if "week_num" not in df_tfc.columns:
        if "week_start" in df_tfc.columns:
            df_tfc["week_num"] = pd.to_datetime(df_tfc["week_start"], errors="coerce").dt.isocalendar().week.astype(
                "Int64")
        else:
            raise KeyError(f"No 'week_num' column found in {tfc_path}")

    # --- Merge TFC (physical fatigue) ---
    # --- Ensure both sides have iso_year and week_num ---
    for d in [df_long, df_tfc]:
        if "iso_year" not in d.columns:
            if "week_start" in d.columns:
                d["iso_year"] = pd.to_datetime(d["week_start"], errors="coerce").dt.isocalendar().year.astype("Int64")
            elif "date" in d.columns:
                d["iso_year"] = pd.to_datetime(d["date"], errors="coerce").dt.isocalendar().year.astype("Int64")
            else:
                raise KeyError(f"No suitable column to derive iso_year in {set(d.columns)}")
        if "week_num" not in d.columns:
            if "week_start" in d.columns:
                d["week_num"] = pd.to_datetime(d["week_start"], errors="coerce").dt.isocalendar().week.astype("Int64")
            elif "date" in d.columns:
                d["week_num"] = pd.to_datetime(d["date"], errors="coerce").dt.isocalendar().week.astype("Int64")
            else:
                raise KeyError(f"No suitable column to derive week_num in {set(d.columns)}")

    # --- Merge TFC (physical fatigue) ---
    df_long = df_long.merge(
        df_tfc[["player_id", "iso_year", "week_num", tfc_col]],
        on=["player_id", "iso_year", "week_num"],
        how="left"
    )
    df_long.rename(columns={tfc_col: "TFC_mean"}, inplace=True)


    df_long = df_long.merge(
        df_sti[["player_id", "iso_year", "week_num", sti_col]],
        on=["player_id", "iso_year", "week_num"],
        how="left"
    )
    df_long.rename(columns={sti_col: "STI_mean"}, inplace=True)

    # Sort per player chronologically
    df_long = df_long.sort_values(["player_id", "date"]).reset_index(drop=True)
    df_long["prev_surface"] = df_long.groupby("player_id")["surface"].shift(1)
    df_long["surf_delta"] = df_long.apply(lambda r: surface_delta(r["prev_surface"], r["surface"]), axis=1)

    # Normalize Transition Cost (TFC)
    df_long["TC"] = df_long["TFC_mean"].fillna(0)
    if df_long["TC"].max() > 0:
        df_long["TC"] = df_long["TC"] / df_long["TC"].max()

    # Surface Adaptation Difficulty
    df_long["SAD"] = df_long["surf_delta"] / 0.8

    # Adaptation Response (rolling win rate)
    df_long["win_flag"] = df_long["label"]
    df_long["AR"] = df_long.groupby("player_id")["win_flag"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Composite TFA
    df_long["TFA"] = 1 - (0.5 * df_long["TC"] + 0.3 * df_long["SAD"]) + 0.2 * df_long["AR"]
    df_long["TFA"] = df_long["TFA"].clip(0, 1)

    return df_long


def weekly_aggregate(df: pd.DataFrame, by_surface=False):
    """Aggregate player-level TFA values to weekly means (optionally surface-specific)."""

    # --- Ensure week_start exists ---
    if "week_start" not in df.columns:
        if {"iso_year", "week_num"}.issubset(df.columns):
            df["week_start"] = df.apply(
                lambda r: datetime.fromisocalendar(
                    int(r["iso_year"]),
                    int(r["week_num"]),
                    1  # Monday
                ),
                axis=1
            )
        else:
            raise KeyError("Cannot derive week_start — missing iso_year/week_num columns.")

    # --- Grouping columns ---
    cols = ["player_id", "iso_year", "week_num"]
    if by_surface:
        cols.append("surface")

    # --- Aggregation ---
    weekly = (
        df.groupby(cols, observed=True)
        .agg(
            week_start=("week_start", "first"),
            matches=("label", "count"),
            win_rate=("label", "mean"),
            TFA_mean=("TFA", "mean"),
        )
        .reset_index()
    )

    return weekly


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Transition Fatigue Adaptation (TFA)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--tfc", required=True)
    ap.add_argument("--sti", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing TFA (physical + surface adaptation)...")
    df_tfa = compute_tfa(master, args.tfc, args.sti)

    match_path = out_root / "tfa_match.csv"
    df_tfa.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level TFA → {match_path} ({len(df_tfa):,} rows)")

    print("[INFO] Aggregating weekly global...")
    weekly = weekly_aggregate(df_tfa, by_surface=False)
    weekly.to_csv(out_root / "tfa_weekly.csv", index=False)
    print(f"[INFO] Saved weekly TFA → {out_root/'tfa_weekly.csv'}")

    print("[INFO] Aggregating weekly by surface...")
    weekly_surf = weekly_aggregate(df_tfa, by_surface=True)
    weekly_surf.to_csv(out_root / "tfa_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface TFA → {out_root/'tfa_weekly_surface.csv'}")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()

import pandas as pd
print(pd.read_csv(r"D:\Tennis\data\indexes\tfc_weekly.csv", nrows=3).columns.tolist())
