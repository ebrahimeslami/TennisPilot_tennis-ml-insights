# -*- coding: utf-8 -*-
"""
Psychological Rivalry Intensity Index (PRII)
--------------------------------------------
Extends RII with psychological context (CPI, FTI, SDI)
and exponential time-decay for older rivalries.

Outputs:
    - prii_match.csv
    - prii_weekly.csv
    - prii_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- helpers ----------
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---------- base rivalry calculation ----------
def compute_rii(df):
    df = df.copy()
    df["pair_id"] = df.apply(lambda r: "_".join(sorted([str(r.winner_id), str(r.loser_id)])), axis=1)

    # frequency
    freq = df["pair_id"].value_counts().to_dict()
    df["freq"] = df["pair_id"].map(freq)
    df["freq_norm"] = df["freq"] / max(df["freq"].max(), 1)

    # balance
    win_counts = df.groupby(["pair_id","winner_id"]).size().unstack(fill_value=0)
    wr = (win_counts.min(axis=1) / win_counts.sum(axis=1)).to_dict()
    df["balance"] = df["pair_id"].map(wr) * 2

    # competitiveness
    df["sets_played"] = df["score"].fillna("").apply(lambda s: len(s.split("-")) if isinstance(s,str) else np.nan)
    df["competitiveness"] = np.clip(df["sets_played"]/3, 0, 1)

    # recency
    latest = df.groupby("pair_id")["tourney_date"].transform("max")
    df["years_since"] = (latest - df["tourney_date"]).dt.days / 365.25
    df["recency"] = np.exp(-df["years_since"]/3)

    # tournament importance
    level_weight = {"G":1.0, "M":0.8, "A":0.6, "B":0.4, "D":0.3, "F":0.2}
    df["tourney_level_norm"] = df["tourney_level"].map(level_weight).fillna(0.3)

    # baseline RII
    wF,wC,wB,wR,wT = 0.25,0.25,0.25,0.15,0.10
    df["RII_base"] = (wF*df["freq_norm"] + wC*df["competitiveness"] +
                      wB*df["balance"] + wR*df["recency"] + wT*df["tourney_level_norm"]).clip(0,1)

    # time-decay factor (fades with age of rivalry)
    df["years_old"] = (datetime.now() - df["tourney_date"]).dt.days / 365.25
    df["decay"] = np.exp(-df["years_old"]/5)   # half-life ≈ 3.5 years
    df["RII"] = df["RII_base"] * df["decay"]

    return df

# ---------- merge psychological context ----------
def merge_context(df_long, fti_path, sdi_path, cpi_path):
    def load_context(path):
        df = pd.read_csv(path)
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
            df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
            df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
        elif "year" in df.columns:
            df["iso_year"] = df["year"].astype(int)
            df["week_num"] = 1
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
            df["week_num"] = df["date"].dt.isocalendar().week.astype(int)
        return df

    df_fti = load_context(fti_path)
    df_sdi = load_context(sdi_path)
    df_cpi = load_context(cpi_path)

    def detect_value_column(df, priority):
        for p in priority:
            if p in df.columns:
                return p
        num_cols = df.select_dtypes(include=[np.number]).columns
        return num_cols[-1] if len(num_cols)>0 else df.columns[-1]

    col_fti = detect_value_column(df_fti, ["FTI","FTI_mean"])
    col_sdi = detect_value_column(df_sdi, ["SDI","SDI_mean"])
    col_cpi = detect_value_column(df_cpi, ["CPI","CPI_weighted"])

    def safe_merge(base, add_df, key_side, value_col, label):
        # rename and drop duplicates
        add_df = add_df.rename(columns={value_col: f"{label}_{key_side}"})
        cols_keep = ["player_id","iso_year","week_num",f"{label}_{key_side}"]
        add_df = add_df[[c for c in cols_keep if c in add_df.columns]].drop_duplicates()
        return base.merge(
            add_df,
            left_on=[f"{key_side}_id","iso_year","week_num"],
            right_on=["player_id","iso_year","week_num"],
            how="left"
        ).drop(columns=["player_id"], errors="ignore")

    for label, dfv, col in [("FTI",df_fti,col_fti),("SDI",df_sdi,col_sdi),("CPI",df_cpi,col_cpi)]:
        df_long = safe_merge(df_long, dfv, "winner", col, label)
        df_long = safe_merge(df_long, dfv, "loser", col, label)

    return df_long

# ---------- compute PRII ----------
def compute_prii(df):
    df = compute_rii(df)
    df["d_CPI"] = (df["CPI_winner"] - df["CPI_loser"]).abs()
    df["d_FTI"] = (df["FTI_winner"] - df["FTI_loser"]).abs()
    df["d_SDI"] = (df["SDI_winner"] - df["SDI_loser"]).abs()

    df["psi"] = 1 + 0.3*df["d_CPI"].fillna(0) + 0.2*df["d_FTI"].fillna(0) - 0.2*df["d_SDI"].fillna(0)
    df["psi"] = df["psi"].clip(0.5, 1.8)

    df["PRII"] = (df["RII"] * df["psi"]).clip(0, 2)
    return df

# ---------- weekly aggregation ----------
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("PRII","count"),
            avg_RII=("RII","mean"),
            avg_PRII=("PRII","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute Psychological Rivalry Intensity Index (PRII) with time decay")
    ap.add_argument("--master", required=True)
    ap.add_argument("--fti", required=True)
    ap.add_argument("--sdi", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master["surface_c"] = master["surface"].map(canon_surface)
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    print("[INFO] Merging context indices (FTI, SDI, CPI)...")
    master = merge_context(master, args.fti, args.sdi, args.cpi)

    print("[INFO] Computing PRII with time decay...")
    df_prii = compute_prii(master)

    path_match = f"{args.out_root}/prii_match.csv"
    df_prii.to_csv(path_match, index=False)
    print(f"[INFO] Saved match-level PRII → {path_match}")

    weekly = weekly_aggregate(df_prii, by_surface=False)
    weekly_path = f"{args.out_root}/prii_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly PRII → {weekly_path}")

    weekly_surf = weekly_aggregate(df_prii, by_surface=True)
    weekly_surf_path = f"{args.out_root}/prii_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface PRII → {weekly_surf_path}")

if __name__ == "__main__":
    main()
