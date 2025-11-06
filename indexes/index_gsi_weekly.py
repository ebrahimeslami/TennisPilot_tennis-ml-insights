import pandas as pd
import numpy as np
import argparse

# ==================================================
# Utility
# ==================================================
def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

def load_index(path, value_col):
    """Flexible loader for sub-indices (FTI, MTI, LII)."""
    df = pd.read_csv(path)
    candidates = [value_col, f"avg_{value_col}", f"{value_col}_mean", f"{value_col}_weighted"]
    found = next((c for c in candidates if c in df.columns), None)
    if not found:
        raise ValueError(f"[ERROR] Could not find column for {value_col} in {path}. Columns: {df.columns.tolist()}")
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
# Compute Greatness Stability Index (GSI)
# ==================================================
def compute_gsi(fti, mti, lii):
    df = fti.merge(mti, on=["player_id","iso_year","week_num"], how="outer", suffixes=("_fti","_mti"))
    df = df.merge(lii, on=["player_id","iso_year","week_num"], how="left")

    if "player_name_x" in df.columns:
        df["player_name"] = df["player_name_x"].combine_first(df.get("player_name_y"))
    else:
        df["player_name"] = df.get("player_name", "Unknown")

    for col in ["FTI","MTI","LII"]:
        df[col] = normalize_series(df.get(col, pd.Series(dtype=float)).fillna(0))

    # Rolling variance of (FTI + MTI)
    df["form_signal"] = (df["FTI"] + df["MTI"]) / 2
    df["variance"] = df.groupby("player_id")["form_signal"].transform(lambda x: x.rolling(12, min_periods=4).var().fillna(0))

    # Compute Greatness Stability Index
    df["GSI_raw"] = 0.4*df["FTI"] + 0.3*df["MTI"] + 0.2*df["LII"] - 0.1*df["variance"]
    df["GSI"] = normalize_series(df.groupby("player_id")["GSI_raw"].transform(lambda x: x.ewm(span=8,min_periods=3).mean()))

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
               avg_GSI=("GSI","mean"),
               matches=("GSI","count"))
          .reset_index()
    )
    return agg

# ==================================================
# Player Career Summary
# ==================================================
def summarize_players(df):
    """Summarize long-term stability for each player."""
    summary = (
        df.groupby("player_id", observed=True)
          .agg(
              player_name=("player_name","last"),
              avg_GSI=("GSI","mean"),
              peak_GSI=("GSI","max"),
              gsi_90th=("GSI", lambda x: np.nanpercentile(x, 90)),
              variance_mean=("variance","mean"),
              first_year=("iso_year","min"),
              last_year=("iso_year","max")
          )
          .reset_index()
    )
    summary["career_span_yrs"] = summary["last_year"] - summary["first_year"] + 1
    summary["stability_score"] = 1 - normalize_series(summary["variance_mean"])
    summary["Greatness_Stability"] = (
        0.6*summary["gsi_90th"] +
        0.25*summary["stability_score"] +
        0.15*summary["avg_GSI"]
    )
    return summary.sort_values("Greatness_Stability", ascending=False)

# ==================================================
# Main
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Compute Greatness Stability Index (GSI)")
    parser.add_argument("--fti", required=True)
    parser.add_argument("--mti", required=True)
    parser.add_argument("--lii", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    print("[INFO] Loading sub-indices...")
    fti = load_index(args.fti, "FTI")
    mti = load_index(args.mti, "MTI")
    lii = load_index(args.lii, "LII")

    print("[INFO] Computing Greatness Stability Index...")
    df_gsi = compute_gsi(fti, mti, lii)
    df_gsi.to_csv(f"{args.out_root}/gsi_match.csv", index=False)
    print(f"[INFO] Saved match-level GSI → {args.out_root}/gsi_match.csv")

    weekly = weekly_aggregate(df_gsi)
    weekly.to_csv(f"{args.out_root}/gsi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly GSI → {args.out_root}/gsi_weekly.csv")

    if "surface_c" in df_gsi.columns:
        weekly_surf = weekly_aggregate(df_gsi, by_surface=True)
        weekly_surf.to_csv(f"{args.out_root}/gsi_weekly_surface.csv", index=False)
        print(f"[INFO] Saved surface GSI → {args.out_root}/gsi_weekly_surface.csv")

    print("[INFO] Generating career stability summary...")
    summary = summarize_players(df_gsi)
    summary.to_csv(f"{args.out_root}/gsi_player_summary.csv", index=False)
    print(f"[INFO] Saved player summary → {args.out_root}/gsi_player_summary.csv")

    print(f"[INFO] Coverage: {df_gsi['iso_year'].min()}–{df_gsi['iso_year'].max()} | Players: {df_gsi['player_id'].nunique()}")

if __name__ == "__main__":
    main()
