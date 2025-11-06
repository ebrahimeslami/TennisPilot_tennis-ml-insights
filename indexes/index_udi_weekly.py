# -*- coding: utf-8 -*-
"""
Upset Dynamics Index (UDI)
--------------------------
Integrates Upset Probability (UPI) and Upset Risk (URI)
to quantify both offensive and defensive upset tendencies.

Outputs:
    - udi_match.csv
    - udi_weekly.csv
    - udi_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ---- helper functions ----
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---- build match-level long format ----
def build_long(master):
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["iso_year"] = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"] = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time
    return m

# ---- compute UDI ----
def compute_udi(df, alpha=1.5):
    df = df.copy()

    # baseline expected probabilities
    def logistic(x): return 1 / (1 + np.exp(alpha * x))

    df["rank_gap"] = np.log(df["winner_rank"] + 1) - np.log(df["loser_rank"] + 1)
    df["p_expected_winner"] = logistic(df["rank_gap"])
    df["p_expected_loser"] = 1 - df["p_expected_winner"]

    # upset flag (winner lower-ranked)
    df["upset_flag"] = (df["winner_rank"] > df["loser_rank"]).astype(int)

    # placeholder contextual modifiers
    df["delta_surface"] = 0.0
    df["delta_momentum"] = 0.0
    df["delta_fatigue"] = 0.0
    df["delta_pressure"] = 0.0

    df["Adj"] = 1 + 0.3*df["delta_surface"] + 0.25*df["delta_momentum"] + \
                   0.25*df["delta_fatigue"] + 0.2*df["delta_pressure"]

    # offensive (upset producer)
    df["UPI"] = (df["Adj"] * df["p_expected_winner"]).clip(0, 1)

    # defensive (upset risk)
    df["URI"] = (df["Adj"] * df["p_expected_loser"]).clip(0, 1)

    # combined volatility
    df["UDI"] = 0.5 * (df["UPI"] + df["URI"])
    return df

# ---- weekly aggregation ----
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")

    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("UDI","count"),
            upset_wins=("upset_flag","sum"),
            UPI_mean=("UPI","mean"),
            URI_mean=("URI","mean"),
            UDI=("UDI","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Compute Upset Dynamics Index (UPI+URI unified)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alpha", type=float, default=1.5)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing UDI...")
    df_udi = compute_udi(build_long(master), alpha=args.alpha)

    path_match = f"{args.out_root}/udi_match.csv"
    df_udi.to_csv(path_match, index=False)
    print(f"[INFO] Saved match-level UDI → {path_match} ({len(df_udi):,} rows)")

    weekly = weekly_aggregate(df_udi, by_surface=False)
    weekly_path = f"{args.out_root}/udi_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly UDI → {weekly_path}")

    weekly_surf = weekly_aggregate(df_udi, by_surface=True)
    weekly_surf_path = f"{args.out_root}/udi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface UDI → {weekly_surf_path}")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly.iso_year.min())}-{int(weekly.iso_year.max())} | Players: {weekly.winner_id.nunique()}")

if __name__ == "__main__":
    main()
