import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# -------------------------
# Tournament Category Weights
# -------------------------
LEVEL_WEIGHTS = {
    "G": 4.0,   # Grand Slam
    "M": 3.0,   # Masters 1000 / WTA 1000
    "A": 2.0,   # ATP/WTA 500
    "B": 1.0,   # ATP/WTA 250
    "C": 1.0,   # Challenger / ITF
    "O": 1.5,   # Olympics / Team events
}

# -------------------------
# Helper Functions
# -------------------------

def normalize_levels(df):
    """Standardize tournament levels across years."""
    df["tourney_level_norm"] = df["tourney_level"].astype(str).str[0].map(LEVEL_WEIGHTS)
    df["tourney_level_norm"].fillna(1.0, inplace=True)
    return df


def long_format(df):
    """Expand winner and loser data into player-level observations."""
    W = df[[
        "tourney_id", "tourney_date", "surface", "tourney_level_norm",
        "winner_id", "winner_name", "winner_ioc"
    ]].copy()
    W["label"] = 1

    L = df[[
        "tourney_id", "tourney_date", "surface", "tourney_level_norm",
        "loser_id", "loser_name", "loser_ioc"
    ]].copy()
    L["label"] = 0

    W.rename(columns={"winner_id": "player_id", "winner_name": "player_name", "winner_ioc": "ioc"}, inplace=True)
    L.rename(columns={"loser_id": "player_id", "loser_name": "player_name", "loser_ioc": "ioc"}, inplace=True)

    long_df = pd.concat([W, L], ignore_index=True)
    long_df = long_df.dropna(subset=["player_id", "tourney_date"])
    long_df["tourney_date"] = pd.to_datetime(long_df["tourney_date"].astype(str), errors="coerce")
    long_df["year"] = long_df["tourney_date"].dt.year
    long_df["week"] = long_df["tourney_date"].dt.isocalendar().week
    long_df["iso_year"] = long_df["tourney_date"].dt.isocalendar().year
    return long_df


def compute_tcs_per_player(df):
    """Compute specialization metrics per player-week."""
    records = []
    for (pid, year, week), sub in tqdm(df.groupby(["player_id", "iso_year", "week"], sort=False), desc="Computing TCS"):
        if sub.empty:
            continue

        # Win ratio per level
        wr = sub.groupby("tourney_level_norm")["label"].mean()
        wr_std = wr.std() if len(wr) > 1 else 0.0

        # Match share per level
        counts = sub["tourney_level_norm"].value_counts(normalize=True)
        entropy = -np.sum(counts * np.log(counts + 1e-9))

        # Normalize entropy (0-1)
        entropy_norm = entropy / np.log(len(counts)) if len(counts) > 1 else 0

        # Final TCS index (higher = more specialization)
        tcs = (1 - entropy_norm) + 0.5 * wr_std

        records.append({
            "player_id": pid,
            "player_name": sub["player_name"].iloc[0],
            "iso_year": year,
            "week": week,
            "matches": len(sub),
            "TCS": round(tcs, 4),
            "Entropy": round(entropy_norm, 4),
            "WR_std": round(wr_std, 4)
        })
    return pd.DataFrame(records)


def compute_tcs_per_surface(df):
    """Surface-specific specialization."""
    records = []
    for (pid, surface, year, week), sub in tqdm(df.groupby(["player_id", "surface", "iso_year", "week"], sort=False), desc="Computing TCS per surface"):
        if sub.empty:
            continue

        wr = sub.groupby("tourney_level_norm")["label"].mean()
        wr_std = wr.std() if len(wr) > 1 else 0.0
        counts = sub["tourney_level_norm"].value_counts(normalize=True)
        entropy = -np.sum(counts * np.log(counts + 1e-9))
        entropy_norm = entropy / np.log(len(counts)) if len(counts) > 1 else 0
        tcs = (1 - entropy_norm) + 0.5 * wr_std

        records.append({
            "player_id": pid,
            "player_name": sub["player_name"].iloc[0],
            "surface": surface,
            "iso_year": year,
            "week": week,
            "matches": len(sub),
            "TCS_surface": round(tcs, 4),
            "Entropy_surface": round(entropy_norm, 4),
            "WR_std_surface": round(wr_std, 4)
        })
    return pd.DataFrame(records)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute Tournament Category Specialization (TCS) Index")
    parser.add_argument("--master", required=True, help="Path to master parquet (e.g. tennis_master_1991.parquet)")
    parser.add_argument("--out_root", required=True, help="Output folder for CSV files")
    args = parser.parse_args()

    print(f"[INFO] Loading master → {args.master}")
    df = pd.read_parquet(args.master)
    df = normalize_levels(df)
    long_df = long_format(df)

    print("[INFO] Saving match-level dataset...")
    match_out = Path(args.out_root) / "tcs_match.csv"
    long_df.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level → {match_out} ({len(long_df):,} rows)")

    # Compute weekly specialization
    print("[INFO] Computing weekly TCS...")
    weekly_df = compute_tcs_per_player(long_df)
    weekly_out = Path(args.out_root) / "tcs_weekly.csv"
    weekly_df.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly TCS → {weekly_out} ({len(weekly_df):,} rows)")

    # Compute surface-specific specialization
    print("[INFO] Computing weekly surface TCS...")
    surf_df = compute_tcs_per_surface(long_df)
    surf_out = Path(args.out_root) / "tcs_weekly_surface.csv"
    surf_df.to_csv(surf_out, index=False)
    print(f"[INFO] Saved weekly surface TCS → {surf_out} ({len(surf_df):,} rows)")

    print(f"[INFO] Coverage: {long_df['year'].min()}–{long_df['year'].max()} | Players: {long_df['player_id'].nunique()}")


if __name__ == "__main__":
    main()
