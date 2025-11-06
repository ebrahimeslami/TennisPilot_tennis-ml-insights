import pandas as pd
import numpy as np
import argparse

# ==============================================
# Utility Functions
# ==============================================
def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

def load_index(path, value_col):
    """Load index file and auto-detect numeric column."""
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

# ==============================================
# Compute Rivalry Exposure Proxy
# ==============================================
def build_rivalry_proxy(master_df):
    """Proxy: fraction of matches vs top-20 opponents."""
    df = master_df.copy()
    df["iso_year"] = pd.to_datetime(df["tourney_date"], errors="coerce").dt.isocalendar().year.astype(int)
    df["week_num"] = pd.to_datetime(df["tourney_date"], errors="coerce").dt.isocalendar().week.astype(int)
    df["is_top20_opp"] = df["loser_rank"].apply(lambda r: 1 if r <= 20 else 0)
    w = df.groupby(["winner_id","iso_year","week_num"]).agg(
        player_name=("winner_name","last"),
        matches=("is_top20_opp","count"),
        top20_matches=("is_top20_opp","sum")
    ).reset_index()
    w["RIVALRY"] = (w["top20_matches"]/w["matches"]).fillna(0)
    w = w.rename(columns={"winner_id":"player_id"})
    return w[["player_id","player_name","iso_year","week_num","RIVALRY"]]

# ==============================================
# Compute Market Value Index (MVI)
# ==============================================
def compute_mvi(fti, pwi, mti, rivalry):
    df = fti.merge(pwi, on=["player_id","iso_year","week_num"], how="outer", suffixes=("_fti","_pwi"))
    df = df.merge(mti, on=["player_id","iso_year","week_num"], how="left")
    df = df.merge(rivalry, on=["player_id","iso_year","week_num"], how="left")

    if "player_name_x" in df.columns:
        df["player_name"] = df["player_name_x"].combine_first(df.get("player_name_y"))
    elif "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    for col in ["FTI","PWI","MTI","RIVALRY"]:
        df[col] = normalize_series(df.get(col, pd.Series(dtype=float)).fillna(0))

    df["AGE_factor"] = normalize_series(df["iso_year"])
    df["MVI_raw"] = (
        0.35*df["FTI"] + 0.30*df["PWI"] + 0.20*df["MTI"] +
        0.10*df["RIVALRY"] + 0.05*df["AGE_factor"]
    )

    # Smooth via exponential weighted average
    df["MVI"] = df.groupby("player_id")["MVI_raw"].transform(lambda x: x.ewm(span=8,min_periods=3).mean())
    return df

# ==============================================
# Compute Endorsement Momentum Score (EMS)
# ==============================================
def compute_ems(df):
    """Calculate momentum as week-to-week MVI growth rate."""
    def per_player(g):
        g = g.sort_values(["iso_year","week_num"])
        g["MVI_change"] = g["MVI"].diff()
        g["EMS_raw"] = normalize_series(g["MVI_change"].fillna(0))
        g["EMS"] = g["EMS_raw"].ewm(span=4,min_periods=2).mean()
        return g
    return df.groupby("player_id", group_keys=False).apply(per_player)

# ==============================================
# Weekly Aggregation
# ==============================================
def weekly_aggregate(df, by_surface=False):
    keys = ["iso_year","week_num","player_id"]
    if by_surface and "surface_c" in df.columns:
        keys.append("surface_c")
    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"
    agg = (
        df.groupby(keys, observed=True)
          .agg(player_name=("player_name","last"),
               avg_MVI=("MVI","mean"),
               avg_EMS=("EMS","mean"),
               matches=("MVI","count"))
          .reset_index()
    )
    return agg

# ==============================================
# Player Career Summary
# ==============================================
def summarize_players(df):
    """Summarize career-level MVI & EMS performance for each player."""
    summary = (
        df.groupby("player_id", observed=True)
          .agg(
              player_name=("player_name","last"),
              avg_MVI=("MVI","mean"),
              max_MVI=("MVI","max"),
              avg_EMS=("EMS","mean"),
              max_EMS=("EMS","max"),
              weeks_above_075=("MVI", lambda x: (x >= 0.75).sum()),
              first_year=("iso_year","min"),
              last_year=("iso_year","max")
          )
          .reset_index()
    )
    summary["career_span_yrs"] = summary["last_year"] - summary["first_year"] + 1
    summary["market_consistency"] = summary["weeks_above_075"] / summary["career_span_yrs"].replace(0, np.nan)
    summary["Legacy_Score"] = (
        0.5*summary["avg_MVI"] +
        0.3*summary["max_MVI"] +
        0.15*summary["avg_EMS"] +
        0.05*summary["market_consistency"]
    )
    return summary.sort_values("Legacy_Score", ascending=False)

# ==============================================
# Main
# ==============================================
def main():
    parser = argparse.ArgumentParser(description="Compute Market Value Index (MVI) + Endorsement Momentum Score (EMS)")
    parser.add_argument("--fti", required=True)
    parser.add_argument("--pwi", required=True)
    parser.add_argument("--mti", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    print("[INFO] Loading input indices...")
    fti = load_index(args.fti, "FTI")
    pwi = load_index(args.pwi, "PWI")
    mti = load_index(args.mti, "MTI")

    print("[INFO] Building rivalry proxy...")
    master_df = pd.read_parquet(args.master) if args.master.endswith(".parquet") else pd.read_csv(args.master)
    rivalry = build_rivalry_proxy(master_df)

    print("[INFO] Computing Market Value Index...")
    df_mvi = compute_mvi(fti, pwi, mti, rivalry)
    df_mvi.to_csv(f"{args.out_root}/mvi_match.csv", index=False)
    print(f"[INFO] Saved match-level MVI → {args.out_root}/mvi_match.csv")

    print("[INFO] Computing Endorsement Momentum Score (EMS)...")
    df_mvi_ems = compute_ems(df_mvi)
    df_mvi_ems.to_csv(f"{args.out_root}/mvi_ems_weekly.csv", index=False)
    print(f"[INFO] Saved EMS → {args.out_root}/mvi_ems_weekly.csv")

    print("[INFO] Aggregating weekly data...")
    weekly = weekly_aggregate(df_mvi_ems)
    weekly.to_csv(f"{args.out_root}/mvi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly MVI → {args.out_root}/mvi_weekly.csv")

    if "surface_c" in df_mvi_ems.columns:
        weekly_surf = weekly_aggregate(df_mvi_ems, by_surface=True)
        weekly_surf.to_csv(f"{args.out_root}/mvi_weekly_surface.csv", index=False)
        print(f"[INFO] Saved surface MVI → {args.out_root}/mvi_weekly_surface.csv")

    print("[INFO] Generating career summary...")
    summary = summarize_players(df_mvi_ems)
    summary.to_csv(f"{args.out_root}/mvi_player_summary.csv", index=False)
    print(f"[INFO] Saved player summary → {args.out_root}/mvi_player_summary.csv")

    print(f"[INFO] Coverage: {df_mvi_ems['iso_year'].min()}–{df_mvi_ems['iso_year'].max()} | Players: {df_mvi_ems['player_id'].nunique()}")

if __name__ == "__main__":
    main()
