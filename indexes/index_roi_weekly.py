import pandas as pd
import numpy as np
import argparse

# ==================================================
# Utility Functions
# ==================================================
def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

def load_index(path, value_col):
    """Generic loader for sub-indices."""
    df = pd.read_csv(path)
    candidates = [value_col, f"avg_{value_col}", f"{value_col}_mean", f"{value_col}_weighted"]
    found = next((c for c in candidates if c in df.columns), None)
    if not found:
        raise ValueError(f"[ERROR] Could not find {value_col} in {path}. Columns: {df.columns.tolist()}")
    df = df.rename(columns={found: value_col})
    if "iso_year" not in df or "week_num" not in df:
        if "week_start" in df:
            dt = pd.to_datetime(df["week_start"], errors="coerce")
            df["iso_year"] = dt.dt.isocalendar().year.astype(int)
            df["week_num"] = dt.dt.isocalendar().week.astype(int)
    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"
    return df[["player_id","player_name","iso_year","week_num",value_col]]

# ==================================================
# ROI Computation
# ==================================================
def compute_roi(fti, mvi, sdi, ems):
    df = fti.merge(mvi, on=["player_id","iso_year","week_num"], how="outer", suffixes=("_fti","_mvi"))
    df = df.merge(sdi, on=["player_id","iso_year","week_num"], how="left")
    df = df.merge(ems, on=["player_id","iso_year","week_num"], how="left")

    if "player_name_x" in df.columns:
        df["player_name"] = df["player_name_x"].combine_first(df.get("player_name_y"))
    else:
        df["player_name"] = df.get("player_name", "Unknown")

    for col in ["FTI","MVI","SDI","EMS"]:
        if col in df.columns:
            df[col] = normalize_series(df[col].fillna(0))
        else:
            df[col] = 0

    # Rolling volatility of FTI or MVI
    df["volatility"] = df.groupby("player_id")["MVI"].transform(lambda x: x.rolling(8, min_periods=4).std().fillna(0.05))

    # ROI Formula
    df["ROI_raw"] = (
        0.4*df["FTI"] + 0.35*df["MVI"] +
        0.15*(1 - df["SDI"]) + 0.10*df["EMS"]
    ) / (1 + df["volatility"])

    df["ROI"] = normalize_series(df.groupby("player_id")["ROI_raw"].transform(lambda x: x.ewm(span=8,min_periods=3).mean()))
    return df

# ==================================================
# Weekly Aggregation
# ==================================================
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","player_id"]
    if by_surface and "surface_c" in df.columns:
        keys.append("surface_c")
    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"
    agg = (
        df.groupby(keys, observed=True)
          .agg(player_name=("player_name","last"),
               avg_ROI=("ROI","mean"),
               matches=("ROI","count"))
          .reset_index()
    )
    return agg

# ==================================================
# Main
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Compute ROI Potential Index (ROI)")
    parser.add_argument("--fti", required=True)
    parser.add_argument("--mvi", required=True)
    parser.add_argument("--sdi", required=True)
    parser.add_argument("--ems", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    print("[INFO] Loading sub-indices...")
    fti = load_index(args.fti, "FTI")
    mvi = load_index(args.mvi, "MVI")
    sdi = load_index(args.sdi, "SDI")
    ems = load_index(args.ems, "EMS")

    print("[INFO] Computing ROI Potential Index...")
    df_roi = compute_roi(fti, mvi, sdi, ems)
    df_roi.to_csv(f"{args.out_root}/roi_match.csv", index=False)
    print(f"[INFO] Saved match-level ROI → {args.out_root}/roi_match.csv")

    weekly = weekly_aggregate(df_roi)
    weekly.to_csv(f"{args.out_root}/roi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly ROI → {args.out_root}/roi_weekly.csv")

    if "surface_c" in df_roi.columns:
        weekly_surf = weekly_aggregate(df_roi, by_surface=True)
        weekly_surf.to_csv(f"{args.out_root}/roi_weekly_surface.csv", index=False)
        print(f"[INFO] Saved surface ROI → {args.out_root}/roi_weekly_surface.csv")

    print(f"[INFO] Coverage: {df_roi['iso_year'].min()}–{df_roi['iso_year'].max()} | Players: {df_roi['player_id'].nunique()}")

if __name__ == "__main__":
    main()
