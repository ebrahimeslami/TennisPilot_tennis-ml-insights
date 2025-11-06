import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- Helpers ----------
def to_datetime_yyyymmdd(series):
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    t = s.strip().lower()
    if "hard" in t: return "Hard"
    if "clay" in t: return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t or "indoor" in t: return "Carpet"
    return "Other"

def build_long(master: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level master data to long-format player-level dataset
    suitable for surface transition or time-based analyses.

    Each match is expanded to two player records (winner + loser),
    carrying date, surface, and week metadata.

    Returns:
        DataFrame with columns:
            player_id, player_name, date, surface, iso_year,
            week_num, week_start, label (1=win, 0=loss)
    """

    # --- Clean and parse tournament dates ---
    master = master.copy()
    master["tourney_date"] = to_datetime_yyyymmdd(master["tourney_date"])
    master = master.dropna(subset=["tourney_date"])
    master = master[master["tourney_date"].dt.year >= 1991].copy()

    # --- Canonical surface label ---
    master["surface_c"] = master["surface"].map(canon_surface)

    # --- Temporal metadata (for weekly aggregations) ---
    master["iso_year"] = master["tourney_date"].dt.isocalendar().year.astype(int)
    master["week_num"] = master["tourney_date"].dt.isocalendar().week.astype(int)
    master["week_start"] = master["tourney_date"].dt.to_period("W-MON").dt.start_time

    # --- Winner records ---
    W = pd.DataFrame({
        "player_id": master["winner_id"].astype(str),
        "player_name": master["winner_name"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "label": 1,  # win
    })

    # --- Loser records ---
    L = pd.DataFrame({
        "player_id": master["loser_id"].astype(str),
        "player_name": master["loser_name"],
        "date": master["tourney_date"],
        "surface": master["surface_c"],
        "iso_year": master["iso_year"],
        "week_num": master["week_num"],
        "week_start": master["week_start"],
        "label": 0,  # loss
    })

    # --- Combine and sort ---
    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["player_id", "date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # --- Sanity check ---
    if df["iso_year"].isna().any():
        df = df[df["iso_year"].notna()].copy()

    return df



# ---------- Baseline 52w ----------
def compute_surface_baseline_52w(df):
    out = []
    for (pid, surf), sub in df.groupby(["player_id","surface"], sort=False):
        s = sub.sort_values("date").set_index("date")
        s["label_prior"] = s["label"].shift(1)
        s["baseline_52w_surface"] = s["label_prior"].rolling("365D", min_periods=5).mean()
        out.append(s.reset_index())
    df2 = pd.concat(out, ignore_index=True)

    # Player-wide fallback
    tmp = []
    for pid, sub in df2.groupby("player_id", sort=False):
        s = sub.sort_values("date").set_index("date")
        s["label_prior_all"] = s["label"].shift(1)
        s["baseline_52w_all"] = s["label_prior_all"].rolling("365D", min_periods=8).mean()
        tmp.append(s.reset_index())
    df2 = pd.concat(tmp, ignore_index=True)
    df2["baseline_winrate"] = df2["baseline_52w_surface"].fillna(df2["baseline_52w_all"]).fillna(0.5)
    return df2


# ---------- Detect transitions ----------
def mark_transitions(df, max_seq=3, max_gap_days=45):
    df["prev_surface"] = df.groupby("player_id")["surface"].shift(1)
    df["prev_date"] = df.groupby("player_id")["date"].shift(1)
    df["days_since_prev"] = (df["date"] - df["prev_date"]).dt.days
    df["switch_flag"] = (
        df["prev_surface"].notna() &
        (df["surface"] != df["prev_surface"]) &
        (df["days_since_prev"] <= max_gap_days)
    ).astype(int)

    df["transition_seq"] = 0
    for pid, sub in df.groupby("player_id", sort=False):
        seq = 0
        last_surface = None
        last_switch_idx = -1
        for idx in sub.sort_values("date").index:
            cur = df.at[idx, "surface"]
            switch = df.at[idx, "switch_flag"]
            if switch == 1:
                seq = 1
                last_surface = cur
                last_switch_idx = idx
            elif (seq > 0) and (cur == last_surface) and (seq < max_seq):
                seq += 1
            else:
                seq = 0
            df.at[idx, "transition_seq"] = seq
    df["transition_flag"] = (df["transition_seq"] > 0).astype(int)
    return df


# ---------- Compute STI ----------
def compute_sti(df):
    df["STI_match"] = np.where(df["transition_flag"] == 1,
                               df["label"] - df["baseline_winrate"], np.nan)
    df["transition_direction"] = np.where(df["switch_flag"] == 1,
                                          df["prev_surface"] + "→" + df["surface"],
                                          np.nan)
    return df


# ---------- Aggregations ----------
def weekly_aggregate(df, by_surface=False):
    cols = ["player_id","player_name","iso_year","week_num"]
    if by_surface:
        cols.append("surface")
    g = df[df["transition_flag"] == 1].groupby(cols, observed=True)
    out = g.agg(STI_mean=("STI_match","mean"), matches=("label","size")).reset_index()
    return out

def direction_aggregate(df):
    dir_df = (
        df[df["transition_flag"] == 1]
        .groupby(["prev_surface","surface","transition_direction"], observed=True)
        .agg(matches=("label","size"), DSTI_mean=("STI_match","mean"))
        .reset_index()
        .sort_values("DSTI_mean", ascending=False)
    )
    return dir_df


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Directional Surface Transition Index (DSTI)")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--k_matches", type=int, default=3)
    ap.add_argument("--gap_days", type=int, default=45)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building long-format player data…")
    df = build_long(master)

    print("[INFO] Computing 52-week baselines…")
    df = compute_surface_baseline_52w(df)

    print("[INFO] Marking transitions…")
    df = mark_transitions(df, max_seq=args.k_matches, max_gap_days=args.gap_days)

    print("[INFO] Computing STI & transition direction…")
    df = compute_sti(df)

    # Save match-level
    match_out = out_root / "sti_directional_match.csv"
    df.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level STI → {match_out} ({len(df):,} rows)")

    # Weekly (global)
    weekly = weekly_aggregate(df, by_surface=False)
    weekly.to_csv(out_root / "sti_directional_weekly.csv", index=False)
    print(f"[INFO] Saved weekly STI → {out_root/'sti_directional_weekly.csv'} ({len(weekly):,} rows)")

    # Weekly (surface)
    weekly_surf = weekly_aggregate(df, by_surface=True)
    weekly_surf.to_csv(out_root / "sti_directional_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface STI → {out_root/'sti_directional_weekly_surface.csv'} ({len(weekly_surf):,} rows)")

    # Directional summary
    dir_df = direction_aggregate(df)
    dir_out = out_root / "sti_transition_direction.csv"
    dir_df.to_csv(dir_out, index=False)
    print(f"[INFO] Saved transition direction summary → {dir_out} ({len(dir_df):,} rows)")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
