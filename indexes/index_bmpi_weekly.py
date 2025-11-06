# -*- coding: utf-8 -*-
"""
Big Match Performance Index (BMPI)
----------------------------------
Measures how well players perform in high-importance matches.

Components:
  - Match Importance Factor (MIF)
  - Performance Quality Factor (PQF)
  - Clutch Enhancement (CE)
  - Composite BMPI = 0.4*PQF + 0.4*CE + 0.2*MIF

Outputs:
  - bmpi_match.csv
  - bmpi_weekly.csv
  - bmpi_weekly_surface.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------
# Helpers
# --------------------------------------------
def to_datetime_yyyymmdd(series):
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    s = s.strip().lower()
    if "hard" in s: return "Hard"
    if "clay" in s: return "Clay"
    if "grass" in s: return "Grass"
    if "carpet" in s or "indoor" in s: return "Carpet"
    return "Other"

def level_weight(level):
    m = {
        "G": 1.0, "M": 0.8, "A": 0.6, "B": 0.4, "D": 0.9,  # Davis Cup / ATP Cup
        "F": 0.7, "L": 0.3
    }
    return m.get(str(level).strip().upper(), 0.5)

def round_weight(r):
    r = str(r).lower()
    if "f" in r and "sf" not in r: return 1.0
    if "sf" in r: return 0.8
    if "qf" in r: return 0.6
    if "r16" in r or "r32" in r: return 0.4
    if "rr" in r: return 0.3
    return 0.2

# --------------------------------------------
# Core computation
# --------------------------------------------
def compute_bmpi(df, cpi_path):
    df_cpi = pd.read_csv(cpi_path)
    df_cpi["player_id"] = df_cpi["player_id"].astype(str)

    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df["surface_c"] = df["surface"].map(canon_surface)

    # Winner + loser expansion
    players = []
    for label in ["winner", "loser"]:
        tmp = pd.DataFrame({
            "player_id": df[f"{label}_id"].astype(str),
            "player_name": df[f"{label}_name"],
            "opp_rank": df["loser_rank"] if label == "winner" else df["winner_rank"],
            "tourney_date": df["tourney_date"],
            "surface": df["surface_c"],
            "tourney_level": df["tourney_level"],
            "round": df["round"],
            "label": 1 if label == "winner" else 0,
            "w_1stWon": df["w_1stWon"],
            "w_2ndWon": df["w_2ndWon"],
            "w_svpt": df["w_svpt"],
            "l_1stWon": df["l_1stWon"],
            "l_2ndWon": df["l_2ndWon"],
            "l_svpt": df["l_svpt"],
        })
        players.append(tmp)

    df_long = pd.concat(players, ignore_index=True).dropna(subset=["player_id", "tourney_date"])

    # Add temporal fields for weekly aggregation
    df_long["iso_year"] = df_long["tourney_date"].dt.isocalendar().year.astype(int)
    df_long["week_num"] = df_long["tourney_date"].dt.isocalendar().week.astype(int)
    df_long["week_start"] = df_long["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Merge CPI (year-based)
    df_long["year"] = df_long["tourney_date"].dt.year
    df_long = df_long.merge(
        df_cpi[["player_id", "year", "CPI_weighted"]],
        on=["player_id", "year"],
        how="left"
    )
    df_long.rename(columns={"CPI_weighted": "CPI"}, inplace=True)

    # Compute BMPI components
    df_long["MIF"] = (
        df_long["tourney_level"].apply(level_weight) * 0.4 +
        df_long["round"].apply(round_weight) * 0.3 +
        (1 - df_long["opp_rank"].fillna(200) / 200) * 0.3
    )
    df_long["MIF"] = df_long["MIF"].clip(0, 1)

    df_long["PQF"] = (
        ((df_long["w_1stWon"] + df_long["w_2ndWon"]) / df_long["w_svpt"]) -
        ((df_long["l_1stWon"] + df_long["l_2ndWon"]) / df_long["l_svpt"])
    ).clip(-1, 1)

    df_long["CE"] = df_long["CPI"].fillna(0) * df_long["MIF"]

    df_long["BMPI"] = 0.4 * df_long["PQF"] + 0.4 * df_long["CE"] + 0.2 * df_long["MIF"]

    return df_long


def weekly_aggregate(df, by_surface=False):
    cols = ["player_id", "player_name", "iso_year", "week_num", "week_start"]
    if by_surface:
        cols.append("surface")
    return (
        df.groupby(cols, observed=True)
        .agg(matches=("BMPI", "size"), BMPI_mean=("BMPI", "mean"))
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )


# --------------------------------------------
# Main
# --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Big Match Performance Index (BMPI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing BMPI...")
    df_bmpi = compute_bmpi(master, args.cpi)

    match_path = out_root / "bmpi_match.csv"
    df_bmpi.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level BMPI → {match_path} ({len(df_bmpi):,} rows)")

    print("[INFO] Aggregating weekly global...")
    weekly = weekly_aggregate(df_bmpi, by_surface=False)
    weekly.to_csv(out_root / "bmpi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly BMPI → {out_root/'bmpi_weekly.csv'}")

    print("[INFO] Aggregating weekly by surface...")
    weekly_surf = weekly_aggregate(df_bmpi, by_surface=True)
    weekly_surf.to_csv(out_root / "bmpi_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface BMPI → {out_root/'bmpi_weekly_surface.csv'}")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
