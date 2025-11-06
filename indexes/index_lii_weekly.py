# -*- coding: utf-8 -*-
"""
Legacy Impact Index (LII)
-------------------------
Quantifies the long-term influence of defining career moments (CDMI)
and how strongly they persist through time, consistency, and prestige.

Outputs:
    - lii_match.csv (event-level memory)
    - lii_weekly.csv
    - lii_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

def to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

# ---- Load helper ----
def load_index(path, value_col, name):
    df = pd.read_csv(path)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)
        df["week_num"] = df["week_start"].dt.isocalendar().week.astype(int)
    if "surface" in df.columns:
        df["surface_c"] = df["surface"].map(canon_surface)
    print(f"[INFO] Loaded {name} ({len(df)} rows)")
    return df.rename(columns={value_col: name})

# ---- Compute LII ----
def compute_lii(cdmi, fti, mti, msi=None):
    print("[INFO] Computing decay and composite legacy weights...")

    current_year = datetime.now().year
    cdmi["decay"] = np.exp(-(current_year - cdmi["iso_year"]) / 5)

    # If tourney_level is missing (weekly), assign mid-prestige
    if "tourney_level" in cdmi.columns:
        cdmi["prestige"] = cdmi["tourney_level"].map(
            {"G":1.0,"M":0.8,"A":0.6,"B":0.4,"D":0.3,"F":0.2}
        ).fillna(0.4)
    else:
        cdmi["prestige"] = 0.5

    # Normalize ID column
    id_col = None
    for col in ["player_id", "winner_id", "loser_id"]:
        if col in cdmi.columns:
            id_col = col
            break
    if id_col is None:
        raise ValueError("No valid player identifier column found in CDMI input.")
    cdmi = cdmi.rename(columns={id_col: "player_id"})

    # Merge consistency (FTI + MTI)
    df = cdmi.merge(
        fti[["player_id","iso_year","week_num","FTI"]],
        on=["player_id","iso_year","week_num"], how="left"
    ).merge(
        mti[["player_id","iso_year","week_num","MTI"]],
        on=["player_id","iso_year","week_num"], how="left"
    )

    df["consistency"] = df[["FTI","MTI"]].mean(axis=1).fillna(0.5)

    # optional media memory (MSI)
    if msi is not None:
        # detect ID and MSI column automatically
        id_col = None
        for c in ["player_id", "winner_id", "loser_id"]:
            if c in msi.columns:
                id_col = c
                break
        val_col = None
        for c in ["avg_MSI", "MSI", "media_mem"]:
            if c in msi.columns:
                val_col = c
                break
        if id_col is None or val_col is None:
            raise ValueError(f"MSI file missing ID or MSI column. Found: {msi.columns.tolist()}")
        msi = msi.rename(columns={id_col: "player_id", val_col: "avg_MSI"})

        df = df.merge(
            msi[["player_id", "iso_year", "week_num", "avg_MSI"]],
            on=["player_id", "iso_year", "week_num"], how="left"
        ).rename(columns={"avg_MSI": "media_mem"})
    else:
        df["media_mem"] = 0.5

    # Surface diversity memory (robust)
    val_col = None
    for c in ["CDMI", "avg_CDMI", "CDMI_weighted"]:
        if c in cdmi.columns:
            val_col = c
            break
    if val_col is None:
        raise ValueError(f"No CDMI-like column found. Columns: {cdmi.columns.tolist()}")

    # Ensure surface column exists
    if "surface_c" not in cdmi.columns:
        cdmi["surface_c"] = "All"

    surf_counts = (
        cdmi.loc[cdmi[val_col].fillna(0) > 0.75]
        .groupby("player_id")["surface_c"]
        .nunique()
        .clip(upper=3)
        .fillna(0)
        .reset_index(name="surf_div")
    )

    df = df.merge(surf_counts, on="player_id", how="left").fillna({"surf_div":0})
    df["surf_div"] /= 3

    # ---- Final LII ----
    # Auto-detect CDMI column
    cdmi_col = None
    for c in ["CDMI", "avg_CDMI", "CDMI_weighted"]:
        if c in df.columns:
            cdmi_col = c
            break
    if cdmi_col is None:
        raise ValueError(f"No CDMI column found in merged DataFrame. Columns: {df.columns.tolist()}")

    wC, wD, wP, wK, wM, wS = 0.35, 0.20, 0.15, 0.15, 0.10, 0.05
    df["LII"] = (
            wC * df[cdmi_col].fillna(0.5) +
            wD * df["decay"] +
            wP * df["prestige"] +
            wK * df["consistency"] +
            wM * df["media_mem"] +
            wS * df["surf_div"]
    ).clip(0, 1)

    return df

# ---- Weekly aggregate ----
def weekly_aggregate(df, by_surface=False):
    # Ensure required columns exist
    if "surface_c" not in df.columns:
        df["surface_c"] = "All"

    keys = ["iso_year", "week_num", "player_id"]
    if by_surface:
        keys.append("surface_c")

    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("player_name", "last"),
            week_start=("week_start", "first"),
            matches=("LII", "count"),
            avg_LII=("LII", "mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Compute Legacy Impact Index (LII)")
    ap.add_argument("--cdmi", required=True)
    ap.add_argument("--fti", required=True)
    ap.add_argument("--mti", required=True)
    ap.add_argument("--msi", required=False)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    cdmi = load_index(args.cdmi, "CDMI", "CDMI")
    fti = load_index(args.fti, "FTI", "FTI")
    mti = load_index(args.mti, "MTI", "MTI")
    msi = load_index(args.msi, "avg_MSI", "MSI") if args.msi else None

    df_lii = compute_lii(cdmi, fti, mti, msi)

    match_path = f"{args.out_root}/lii_match.csv"
    df_lii.to_csv(match_path, index=False)
    print(f"[INFO] Saved match-level LII → {match_path}")

    weekly = weekly_aggregate(df_lii, by_surface=False)
    weekly.to_csv(f"{args.out_root}/lii_weekly.csv", index=False)
    weekly_surf = weekly_aggregate(df_lii, by_surface=True)
    weekly_surf.to_csv(f"{args.out_root}/lii_weekly_surface.csv", index=False)
    print("[INFO] Saved weekly & surface LII")

if __name__ == "__main__":
    main()

