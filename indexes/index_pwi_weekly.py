import pandas as pd
import numpy as np
import argparse

# ==============================================
# Helpers
# ==============================================
def normalize_series(s):
    """Normalize a numeric pandas Series to [0, 1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

def load_index(path, value_col):
    """Load an index file, normalize the main column, and ensure weekly structure."""
    df = pd.read_csv(path)

    # Auto-detect the main numeric column
    candidates = [
        value_col,
        f"{value_col}_mean",
        f"avg_{value_col}",
        f"{value_col}_weighted",
        f"{value_col.lower()}_mean",
        "value",
        "avg_value"
    ]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break

    if found is None:
        raise ValueError(
            f"[ERROR] Could not find any matching column for {value_col} in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={found: value_col})
    df[value_col] = normalize_series(df[value_col])

    # infer temporal keys
    if "iso_year" not in df.columns or "week_num" not in df.columns:
        if "week_start" in df.columns:
            dt = pd.to_datetime(df["week_start"], errors="coerce")
            df["iso_year"] = dt.dt.isocalendar().year.astype(int)
            df["week_num"] = dt.dt.isocalendar().week.astype(int)
        else:
            raise ValueError(f"[ERROR] Cannot infer week/year from {path}")

    keep_cols = ["player_id", "player_name", "iso_year", "week_num", value_col]
    df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=[value_col])
    return df


# ==============================================
# Core Computation
# ==============================================
def compute_pwi(fti, sdi, mti):
    """Compute the Prime Window Index by merging FTI, SDI, and MTI."""
    df = fti.merge(sdi, on=["player_id", "iso_year", "week_num"], how="left", suffixes=("_fti", "_sdi"))
    df = df.merge(mti, on=["player_id", "iso_year", "week_num"], how="left")
    df = df.rename(columns={"FTI": "fti", "SDI": "sdi", "MTI": "mti"})

    def per_player(g):
        g = g.sort_values(["iso_year", "week_num"])
        g["F_base"] = g["fti"].rolling(52, min_periods=8).mean()
        g["F_std"] = g["fti"].rolling(52, min_periods=8).std()
        g["PWI_raw"] = (
            0.4 * ((g["fti"] - g["F_base"]) / (g["F_std"] + 1e-6)) +
            0.3 * (1 - g["sdi"].fillna(0)) +
            0.3 * g["mti"].fillna(0)
        )
        g["PWI"] = normalize_series(g["PWI_raw"])
        return g

    return df.groupby("player_id", group_keys=False).apply(per_player)

# ==============================================
# Weekly Aggregation
# ==============================================
def weekly_aggregate(df, by_surface=False):
    """Aggregate PWI values by player-week (and optionally surface)."""
    keys = ["iso_year", "week_num", "player_id"]
    if by_surface and "surface_c" in df.columns:
        keys.append("surface_c")

    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("player_name", "last"),
            avg_PWI=("PWI", "mean"),
            matches=("PWI", "count")
        )
        .reset_index()
    )
    return agg

# ==============================================
# Prime Span Detection (with Ranking)
# ==============================================
def detect_prime_spans(df, threshold=0.75, min_weeks=6):
    """
    Detect continuous prime periods (>= min_weeks, PWI >= threshold)
    and assign a rank for each player based on mean_PWI and duration.
    """
    spans = []
    for pid, g in df.groupby("player_id"):
        g = g.sort_values(["iso_year", "week_num"])
        g["prime_flag"] = g["PWI"] >= threshold
        streak = 0
        start, end = None, None

        for _, row in g.iterrows():
            if row["prime_flag"]:
                if streak == 0:
                    start = row["iso_year"], row["week_num"]
                streak += 1
                end = row["iso_year"], row["week_num"]
            else:
                if streak >= min_weeks:
                    spans.append({
                        "player_id": pid,
                        "player_name": g["player_name"].iloc[0],
                        "start_year": start[0],
                        "start_week": start[1],
                        "end_year": end[0],
                        "end_week": end[1],
                        "duration_weeks": streak,
                        "mean_PWI": g.loc[g["prime_flag"], "PWI"].mean(),
                        "max_PWI": g["PWI"].max()
                    })
                streak = 0

        # trailing streak
        if streak >= min_weeks:
            spans.append({
                "player_id": pid,
                "player_name": g["player_name"].iloc[0],
                "start_year": start[0],
                "start_week": start[1],
                "end_year": end[0],
                "end_week": end[1],
                "duration_weeks": streak,
                "mean_PWI": g.loc[g["prime_flag"], "PWI"].mean(),
                "max_PWI": g["PWI"].max()
            })

    spans_df = pd.DataFrame(spans)
    if spans_df.empty:
        return spans_df

    # rank prime windows per player
    spans_df["prime_rank"] = spans_df.groupby("player_id")["mean_PWI"] \
        .rank(ascending=False, method="dense")
    spans_df.sort_values(["player_id", "prime_rank"], inplace=True)

    return spans_df

# ==============================================
# Main
# ==============================================
def main():
    parser = argparse.ArgumentParser(description="Compute Prime Window Index (PWI) with Prime Span Detection")
    parser.add_argument("--fti", required=True)
    parser.add_argument("--sdi", required=True)
    parser.add_argument("--mti", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading index data...")
    fti = load_index(args.fti, "FTI")
    sdi = load_index(args.sdi, "SDI")
    mti = load_index(args.mti, "MTI")

    print(f"[INFO] Computing Prime Window Index (PWI)...")
    df_pwi = compute_pwi(fti, sdi, mti)

    out_root = args.out_root
    df_pwi.to_csv(f"{out_root}/pwi_match.csv", index=False)
    print(f"[INFO] Saved match-level PWI → {out_root}/pwi_match.csv")

    weekly = weekly_aggregate(df_pwi)
    weekly.to_csv(f"{out_root}/pwi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly PWI → {out_root}/pwi_weekly.csv")

    if "surface_c" in df_pwi.columns:
        weekly_surf = weekly_aggregate(df_pwi, by_surface=True)
        weekly_surf.to_csv(f"{out_root}/pwi_weekly_surface.csv", index=False)
        print(f"[INFO] Saved surface PWI → {out_root}/pwi_weekly_surface.csv")

    print(f"[INFO] Detecting Prime Windows (≥6 consecutive weeks, PWI ≥ 0.75)...")
    spans = detect_prime_spans(df_pwi, threshold=0.75, min_weeks=6)
    spans.to_csv(f"{out_root}/pwi_prime_spans.csv", index=False)
    print(f"[INFO] Saved prime span summary → {out_root}/pwi_prime_spans.csv")

    print(f"[INFO] Coverage: {df_pwi['iso_year'].min()}–{df_pwi['iso_year'].max()} | Players: {df_pwi['player_id'].nunique()}")

if __name__ == "__main__":
    main()
