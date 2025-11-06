import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# ================================
# Helper Functions
# ================================

def to_datetime_yyyymmdd(series):
    """Convert YYYYMMDD integers to datetime."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

def build_long(master):
    """Convert master match-level file into player-level long format."""
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    W = pd.DataFrame({
        "player_id": master["winner_id"].astype(str),
        "player_name": master["winner_name"],
        "rank": master["winner_rank"],
        "date": master["tourney_date"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "surface_c": master.get("surface", "All"),
        "label": 1
    })
    L = pd.DataFrame({
        "player_id": master["loser_id"].astype(str),
        "player_name": master["loser_name"],
        "rank": master["loser_rank"],
        "date": master["tourney_date"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "surface_c": master.get("surface", "All"),
        "label": 0
    })
    df = pd.concat([W, L], ignore_index=True)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df


# ================================
# Core BAI Computation
# ================================

def compute_bai(df):
    """Compute Breakthrough Alert Index per player-week."""
    def per_player(g):
        g = g.sort_values("date")
        g["rank_norm"] = 1 - (g["rank"].fillna(2000) / 2000).clip(0, 1)
        g["perf_4w"] = g["label"].rolling(4, min_periods=2).mean()
        g["perf_8w_std"] = g["label"].rolling(8, min_periods=4).std()
        g["ΔPerf"] = g["perf_4w"].diff()
        g["ΔRank"] = g["rank_norm"].diff()

        # Age effect proxy (peak alert around 24–28 years)
        g["AgeFactor"] = np.exp(-((27 - 24) ** 2) / (2 * 4 ** 2))

        # Z-normalization
        for col in ["ΔPerf", "ΔRank", "perf_8w_std"]:
            g[col + "_z"] = (g[col] - g[col].mean()) / (g[col].std() + 1e-6)

        # Weighted BAI computation
        g["BAI"] = (
            0.4 * g["ΔPerf_z"].fillna(0)
            + 0.3 * g["ΔRank_z"].fillna(0)
            + 0.2 * (1 - g["perf_8w_std_z"].fillna(0))
            + 0.1 * g["AgeFactor"]
        ).clip(0, 1)
        return g

    return df.groupby("player_id", group_keys=False).apply(per_player)


# ================================
# Breakthrough Signal Detection
# ================================

def add_breakthrough_signal(df):
    """Flag weeks where a player shows sustained breakthrough behavior."""
    def per_player(g):
        g = g.sort_values(["iso_year", "week_num"])
        g["Breakthrough_Flag"] = 0
        g["Breakthrough_Signal_Week"] = np.nan

        high = g["BAI"] >= 0.8
        streak = 0
        for i in range(len(g)):
            if high.iloc[i]:
                streak += 1
                if streak >= 2:
                    g.loc[g.index[i], "Breakthrough_Flag"] = 1
                    g.loc[g.index[i], "Breakthrough_Signal_Week"] = g.iloc[i - 1]["week_start"]
            else:
                streak = 0
        return g

    return df.groupby("player_id", group_keys=False).apply(per_player)


# ================================
# Weekly Aggregation
# ================================

def weekly_aggregate(df, by_surface=False):
    """Aggregate BAI by week and optionally by surface."""
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
            matches=("BAI", "count"),
            avg_BAI=("BAI", "mean"),
            breakthrough_weeks=("Breakthrough_Flag", "sum")
        )
        .reset_index()
    )
    return agg


# ================================
# Breakthrough Summary
# ================================

def breakthrough_summary(df):
    """Summarize the first and total breakthroughs per player."""
    def per_player(g):
        g = g.sort_values("date")
        breakthroughs = g[g["Breakthrough_Flag"] == 1]
        if len(breakthroughs) == 0:
            return pd.Series({
                "first_breakthrough_week": np.nan,
                "total_breakthroughs": 0,
                "avg_BAI_at_breakthrough": np.nan,
                "max_BAI": g["BAI"].max(),
                "last_breakthrough_year": np.nan
            })
        first_week = breakthroughs["week_start"].iloc[0]
        return pd.Series({
            "first_breakthrough_week": first_week,
            "total_breakthroughs": len(breakthroughs),
            "avg_BAI_at_breakthrough": breakthroughs["BAI"].mean(),
            "max_BAI": g["BAI"].max(),
            "last_breakthrough_year": breakthroughs["iso_year"].iloc[-1]
        })

    out = df.groupby(["player_id", "player_name"], group_keys=False).apply(per_player).reset_index()
    return out


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)
    df_long = build_long(master)

    print("[INFO] Computing Breakthrough Alert Index (BAI)...")
    df_bai = compute_bai(df_long)
    df_bai = add_breakthrough_signal(df_bai)

    out_root = args.out_root

    # Match-level output
    df_bai.to_csv(f"{out_root}/bai_match.csv", index=False)
    print(f"[INFO] Saved match-level BAI → {out_root}/bai_match.csv")

    # Weekly aggregation
    weekly = weekly_aggregate(df_bai, by_surface=False)
    weekly.to_csv(f"{out_root}/bai_weekly.csv", index=False)
    print(f"[INFO] Saved weekly BAI → {out_root}/bai_weekly.csv")

    # Surface-specific weekly
    weekly_surf = weekly_aggregate(df_bai, by_surface=True)
    weekly_surf.to_csv(f"{out_root}/bai_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface BAI → {out_root}/bai_weekly_surface.csv")

    # Summary report
    summary = breakthrough_summary(df_bai)
    summary.to_csv(f"{out_root}/bai_breakthrough_summary.csv", index=False)
    print(f"[INFO] Saved breakthrough summary → {out_root}/bai_breakthrough_summary.csv")

    print(f"[INFO] Coverage: {df_bai['iso_year'].min()}–{df_bai['iso_year'].max()} | Players: {df_bai['player_id'].nunique()}")

if __name__ == "__main__":
    main()
