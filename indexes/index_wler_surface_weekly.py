"""
index_wler_surface_weekly.py
Author: Ebrahim Eslami
Index: Weak Link Exploitation Rate (WLER)
Purpose:
  Quantify how effectively players exploit opponents’ known weaknesses,
  and compute both global and surface-specific weekly WLER from 1991–2025.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# ------------------ HELPER FUNCTIONS ------------------

def compute_weakness_profiles(df, rolling_window=10):
    """
    Build rolling baseline weakness profiles for each player.
    Columns returned: ['opp_id','date','serve_weak','bp_weak','endurance_weak']
    """
    all_profiles = []
    df = df.sort_values(["player_id", "date"])

    for pid, group in tqdm(df.groupby("player_id"), desc="Building opponent weakness baselines"):
        group = group.sort_values("date").copy()
        group["serve_weak"] = group["opp_2ndWon"].rolling(rolling_window, min_periods=3).mean()
        group["bp_weak"] = (
            group["opp_bpSaved"] / group["opp_bpFaced"].replace(0, np.nan)
        ).rolling(rolling_window, min_periods=3).mean()
        group["endurance_weak"] = (
            group["minutes"] / group["sets"]
        ).rolling(rolling_window, min_periods=3).mean()
        group["opp_id"] = pid
        all_profiles.append(group[["opp_id", "date", "serve_weak", "bp_weak", "endurance_weak"]])

    return pd.concat(all_profiles, ignore_index=True)


def compute_match_exploitation(df, opp_prof):
    """
    Merge player match data with opponent baseline and compute exploitation metrics.
    """
    df = df.merge(opp_prof, on=["opp_id", "date"], how="left")

    df["serve_exploit"] = np.where(
        (df["opp_2ndWon"] < 0.5) & (df["player_return_2ndWon"] > df["opp_2ndWon"]),
        (df["player_return_2ndWon"] - df["opp_2ndWon"]).clip(lower=0), 0,
    )

    df["bp_exploit"] = np.where(
        (df["opp_bpSaved"] / df["opp_bpFaced"].replace(0, np.nan) < 0.6)
        & (df["player_bpConv"] > 0.4),
        (df["player_bpConv"] - (df["opp_bpSaved"] / df["opp_bpFaced"])).clip(lower=0), 0,
    )

    df["endurance_exploit"] = np.where(
        (df["endurance_weak"] < 35) & (df["minutes"] > 120),
        (df["minutes"] / 180).clip(0, 1), 0,
    )

    df["total_exploit"] = df[["serve_exploit", "bp_exploit", "endurance_exploit"]].mean(axis=1)
    return df


def weekly_aggregate(df, surface_specific=False):
    """Aggregate match-level exploitation to weekly per-player WLER."""
    # ✅ Ensure date is datetime before .dt accessor
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.year

    group_cols = ["player_id", "player_name", "year", "week"]
    if surface_specific:
        group_cols.append("surface")

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg({
            "serve_exploit": "mean",
            "bp_exploit": "mean",
            "endurance_exploit": "mean",
            "total_exploit": "mean",
            "label": "count",
        })
        .rename(columns={"label": "matches"})
    )

    # Normalize 0–1 for consistency
    for c in ["serve_exploit", "bp_exploit", "endurance_exploit", "total_exploit"]:
        agg[c] = (agg[c] - agg[c].min()) / (agg[c].max() - agg[c].min() + 1e-9)

    agg["WLER"] = agg["total_exploit"]
    return agg


# ------------------ MAIN EXECUTION ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="Path to master parquet file")
    ap.add_argument("--out_root", required=True, help="Output directory")
    ap.add_argument("--rolling_window", type=int, default=10, help="Rolling matches window for weakness baseline")
    args = ap.parse_args()

    master_path = Path(args.master)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master data → {master_path}")
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), errors="coerce")

    # --- Structure dataset ---
    long_df = pd.DataFrame({
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"],
        "opp_2ndWon": df["l_2ndWon"] / df["l_svpt"],
        "opp_bpSaved": df["l_bpSaved"],
        "opp_bpFaced": df["l_bpFaced"],
        "player_return_2ndWon": df["w_2ndWon"] / df["w_svpt"],
        "player_bpConv": (df["l_bpFaced"] - df["l_bpSaved"]) / df["l_bpFaced"].replace(0, np.nan),
        "minutes": df["minutes"],
        "sets": np.where(df["best_of"] == 5, 5, 3),
        "label": 1,
        "date": df["tourney_date"],
        "surface": df["surface"],
    }).dropna()

    # --- Opponent weakness profiles ---
    opp_prof = compute_weakness_profiles(long_df, rolling_window=args.rolling_window)

    # --- Compute match-level exploitation ---
    df_exp = compute_match_exploitation(long_df, opp_prof)

    # ✅ Convert date to datetime again after merge (fix)
    df_exp["date"] = pd.to_datetime(df_exp["date"], errors="coerce")

    # --- Save match-level results ---
    match_out = out_root / "wler_match.csv"
    df_exp.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level exploitation → {match_out} ({len(df_exp):,} rows)")

    # --- Global weekly WLER ---
    weekly_global = weekly_aggregate(df_exp, surface_specific=False)
    global_out = out_root / "wler_weekly.csv"
    weekly_global.to_csv(global_out, index=False)
    print(f"[INFO] Saved global weekly WLER → {global_out} ({len(weekly_global):,} rows)")

    # --- Surface-specific WLER ---
    weekly_surface = weekly_aggregate(df_exp, surface_specific=True)
    surface_out = out_root / "wler_weekly_surface.csv"
    weekly_surface.to_csv(surface_out, index=False)
    print(f"[INFO] Saved surface-specific weekly WLER → {surface_out} ({len(weekly_surface):,} rows)")

if __name__ == "__main__":
    main()
