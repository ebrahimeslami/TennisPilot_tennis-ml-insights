# -*- coding: utf-8 -*-
"""
Context-Adjusted Upset Dynamics Index (CA-UDI)
----------------------------------------------
Combines UPI+URI logic with contextual modifiers (FTI, SDI, CPI).

Outputs:
    - caudi_match.csv
    - caudi_weekly.csv
    - caudi_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# --- helpers ---
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# --- long format ---
def build_long(master):
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["iso_year"] = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"] = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time
    return m

# --- context merging ---
def merge_context(df_long, fti_path, sdi_path, cpi_path):
    # load weekly datasets
    df_fti = pd.read_csv(fti_path)
    df_sdi = pd.read_csv(sdi_path)
    df_cpi = pd.read_csv(cpi_path)

    # ensure date structure (week_start → iso_year/week_num)
    for df in [df_fti, df_sdi, df_cpi]:
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
            df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
            df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
        elif "year" in df.columns:
            df["iso_year"] = df["year"].astype(int)
            if "week_num" not in df.columns:
                df["week_num"] = 1

    # detect main numeric column automatically
    def detect_value_column(df, priority):
        for p in priority:
            if p in df.columns:
                return p
        return df.select_dtypes(include=[np.number]).columns[-1]

    col_fti = detect_value_column(df_fti, ["FTI", "FTI_mean"])
    col_sdi = detect_value_column(df_sdi, ["SDI", "SDI_mean"])
    col_cpi = detect_value_column(df_cpi, ["CPI", "CPI_weighted"])

    # utility: safe merge with cleanup
    def safe_merge(base, add_df, col, label, side):
        dfm = add_df[["player_id", "iso_year", "week_num", col]].rename(columns={col: f"{label}_{side}"})
        key = "winner_id" if side == "w" else "loser_id"
        merged = (
            base.merge(
                dfm,
                left_on=[key, "iso_year", "week_num"],
                right_on=["player_id", "iso_year", "week_num"],
                how="left"
            )
        )
        # drop redundant columns created by merge
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith("player_id_")])
        return merged

    # apply sequentially
    for label, d, col in [("FTI", df_fti, col_fti), ("SDI", df_sdi, col_sdi), ("CPI", df_cpi, col_cpi)]:
        df_long = safe_merge(df_long, d, col, label, "w")
        df_long = safe_merge(df_long, d, col, label, "l")

    return df_long

# --- compute CA-UDI ---
def compute_caudi(df, alpha=1.5):
    def logistic(x): return 1 / (1 + np.exp(alpha * x))
    df["rank_gap"] = np.log(df["winner_rank"] + 1) - np.log(df["loser_rank"] + 1)
    df["p_expected_winner"] = logistic(df["rank_gap"])
    df["p_expected_loser"] = 1 - df["p_expected_winner"]

    # context deltas
    df["d_fti"] = df.get("FTI_mean", 0) - df.get("FTI_mean_l", 0)
    df["d_sdi"] = df.get("SDI_mean", 0) - df.get("SDI_mean_l", 0)
    df["d_cpi"] = df.get("CPI_weighted", 0) - df.get("CPI_weighted_l", 0)

    df["Adj"] = 1 + 0.25*df["d_fti"].fillna(0) - 0.25*df["d_sdi"].fillna(0) + 0.25*df["d_cpi"].fillna(0)
    df["Adj"] = df["Adj"].clip(0.5, 1.5)

    df["upset_flag"] = (df["winner_rank"] > df["loser_rank"]).astype(int)
    df["UPI_ctx"] = (df["Adj"] * df["p_expected_winner"]).clip(0,1)
    df["URI_ctx"] = (df["Adj"] * df["p_expected_loser"]).clip(0,1)
    df["CA_UDI"] = 0.5 * (df["UPI_ctx"] + df["URI_ctx"])
    return df

# --- weekly aggregation ---
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","winner_id"]
    if by_surface: keys.append("surface_c")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("winner_name","last"),
            week_start=("week_start","first"),
            matches=("CA_UDI","count"),
            upset_wins=("upset_flag","sum"),
            UPI_ctx=("UPI_ctx","mean"),
            URI_ctx=("URI_ctx","mean"),
            CA_UDI=("CA_UDI","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Compute Context-Adjusted Upset Dynamics Index (CA-UDI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--fti", required=True)
    ap.add_argument("--sdi", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alpha", type=float, default=1.5)
    args = ap.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)
    df_long = build_long(master)
    print("[INFO] Merging contextual features (FTI, SDI, CPI)…")
    df_long = merge_context(df_long, args.fti, args.sdi, args.cpi)

    print("[INFO] Computing context-adjusted UDI...")
    df_caudi = compute_caudi(df_long, alpha=args.alpha)

    # save outputs
    match_path = f"{args.out_root}/caudi_match.csv"
    df_caudi.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level CA-UDI → {match_path}")

    weekly = weekly_aggregate(df_caudi, by_surface=False)
    weekly_path = f"{args.out_root}/caudi_weekly.csv"
    weekly.to_csv(weekly_path, index=False)
    print(f"[INFO] Saved weekly CA-UDI → {weekly_path}")

    weekly_surf = weekly_aggregate(df_caudi, by_surface=True)
    weekly_surf_path = f"{args.out_root}/caudi_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_path, index=False)
    print(f"[INFO] Saved weekly surface CA-UDI → {weekly_surf_path}")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly.iso_year.min())}-{int(weekly.iso_year.max())} | Players: {weekly.winner_id.nunique()}")

if __name__ == "__main__":
    main()
