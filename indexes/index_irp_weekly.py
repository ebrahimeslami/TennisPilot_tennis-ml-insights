"""
Injury Risk Predictor (IRP) — Match / Weekly / Weekly-Surface
=============================================================
Goal
----
Estimate per-match injury/withdrawal risk using recent match load, recovery
gaps, match intensity proxies, age, surface, and (optionally) travel fatigue.

Labels
------
Injury/withdrawal event is inferred from the score text containing:
  'RET', 'W/O', 'WO', 'DEF', 'ABN' (case-insensitive).
This is a standard proxy when explicit injury data are unavailable.

Features (per player, at match time)
------------------------------------
- days_since_prev                (recovery gap)
- matches_last_7/14/28          (workload)
- minutes_last_7/14/28          (workload intensity)
- avg_minutes_per_set_last_10   (intensity proxy)
- back_to_back_flag             (days_since_prev <= 1)
- long_match_flag               (>=150min BO3 or >=210min BO5)
- age_at_match                  (winner_age/loser_age as applicable)
- surface_onehot                (Hard, Clay, Grass, Carpet/Indoor/Other)
- TFC_adj_prev                  (OPTIONAL, if --tfc_match is provided)

Model
-----
- If scikit-learn is available, fit a RandomForestClassifier (fast, robust).
- If not, fall back to a calibrated heuristic (sigmoid of z-scored features).

Outputs (CSV)
-------------
- irp_match.csv:
    player_id, player_name, date, iso_year, week_num, week_start, surface,
    features..., injury_label, irp_proba
- irp_weekly.csv:
    player_id, player_name, iso_year, week_num, week_start, matches, irp_mean
- irp_weekly_surface.csv:
    same as weekly + surface

Run
---
python index_irp_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" ^
                           --out_root "D:\\Tennis\\data\\indexes" ^
                           [--tfc_match "D:\\Tennis\\data\\indexes\\tfc_match.csv"]

Notes
-----
- Uses data from 1991+ only (consistent with your master).
- Safe in absence of TFC: the column will be filled with NaN and ignored.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ---------------------- Utils ----------------------

RET_TOKENS = ("RET", "W/O", "WO", "DEF", "ABN")

SURF_CANON = {
    None: "Other",
    "": "Other",
}
# canonicalize surface strings
def _canon_surface(s):
    if not isinstance(s, str):
        return "Other"
    t = s.strip().lower()
    if "hard" in t: return "Hard"
    if "clay" in t: return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t: return "Carpet"
    if "indoor" in t: return "Carpet"
    return "Other"

def to_datetime_yyyymmdd(series):
    s = pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return s

def _is_long_match(minutes, best_of):
    # thresholds: 150min (BO3) or 210min (BO5)
    if pd.isna(minutes) or pd.isna(best_of):
        return False
    if int(best_of) >= 5:
        return minutes >= 210
    return minutes >= 150

def _injury_from_score(score):
    if not isinstance(score, str):
        return 0
    up = score.upper()
    return int(any(tok in up for tok in RET_TOKENS))

def _safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return np.divide(a, b)

# ---------------------- Build player-match ----------------------

def build_player_matches(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()

    # robust dates
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # surface
    df["surface_c"] = df["surface"].map(_canon_surface)

    # common time keys
    df["iso_year"] = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Winner perspective
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "age_at_match": df["winner_age"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "label_win": 1,
        "best_of": df["best_of"],
        "minutes": df["minutes"],
        "score": df["score"],
        "surface": df["surface_c"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "tourney_id": df["tourney_id"],
    })

    # Loser perspective
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "age_at_match": df["loser_age"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "label_win": 0,
        "best_of": df["best_of"],
        "minutes": df["minutes"],
        "score": df["score"],
        "surface": df["surface_c"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "tourney_id": df["tourney_id"],
    })

    PM = pd.concat([W, L], ignore_index=True)
    PM = PM.dropna(subset=["player_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)

    # injury label for CURRENT match
    PM["injury_label"] = PM["score"].apply(_injury_from_score)

    # convenient flags
    PM["long_match_flag"] = [_is_long_match(m, b) for m, b in zip(PM["minutes"], PM["best_of"])]
    PM["long_match_flag"] = PM["long_match_flag"].astype(int)

    return PM

# ---------------------- Feature engineering ----------------------

def add_temporal_features(PM: pd.DataFrame) -> pd.DataFrame:
    PM = PM.copy()

    # days since previous match for player
    PM["prev_date"] = PM.groupby("player_id")["date"].shift(1)
    PM["days_since_prev"] = (PM["date"] - PM["prev_date"]).dt.days
    PM["days_since_prev"] = PM["days_since_prev"].fillna(30)

    # rolling workloads
    for window, tag in [(7, "7"), (14, "14"), (28, "28")]:
        PM[f"matches_last_{tag}"] = (
            PM.groupby("player_id")["date"]
              .rolling(window=200, min_periods=1)  # dummy to enable transform below
              .apply(lambda s: np.nan)  # placeholder; we'll overwrite with vectorized logic
              .reset_index(level=0, drop=True)
        )
    # Vectorized window counts using previous dates
    # For each row i, count prior rows within X days
    arr_dates = PM["date"].values.astype("datetime64[D]")
    arr_pid = PM["player_id"].values
    PM["matches_last_7"] = 0
    PM["matches_last_14"] = 0
    PM["matches_last_28"] = 0
    # two-pointer per player for speed
    for pid, idx in PM.groupby("player_id").indices.items():
        ind = np.arange(idx, idx + PM["player_id"].value_counts()[pid])
        # Actually groupby.indices returns starting index only; change approach:
        pass
# Continue file (place directly after the previous block)

def _count_prior_within_days(sub_dates: pd.Series, days: int) -> np.ndarray:
    # For each index i, count number of prior dates > date_i - days
    vals = sub_dates.values.astype("datetime64[D]").astype("int64")
    out = np.zeros(len(vals), dtype=int)
    j = 0
    for i in range(len(vals)):
        threshold = vals[i] - days
        while j < i and vals[j] <= threshold:
            j += 1
        out[i] = i - j
    return out

def finalize_workload_features(PM: pd.DataFrame) -> pd.DataFrame:
    PM = PM.copy()

    # Ensure 'days_since_prev' exists
    if "days_since_prev" not in PM.columns:
        PM["prev_date"] = PM.groupby("player_id")["date"].shift(1)
        PM["days_since_prev"] = (PM["date"] - PM["prev_date"]).dt.days
        PM["days_since_prev"] = PM["days_since_prev"].fillna(30)

    PM["matches_last_7"] = 0
    PM["matches_last_14"] = 0
    PM["matches_last_28"] = 0
    PM["minutes_last_7"] = 0.0
    PM["minutes_last_14"] = 0.0
    PM["minutes_last_28"] = 0.0
    PM["avg_minutes_per_set_last_10"] = np.nan

    for pid, sub in PM.groupby("player_id", sort=False):
        idx = sub.index

        # Count prior matches within 7/14/28 days (player-local, prior only)
        dates = sub["date"].values.astype("datetime64[D]").astype("int64")
        m7 = np.zeros(len(dates), dtype=int)
        m14 = np.zeros(len(dates), dtype=int)
        m28 = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            cur = dates[i]
            # prior only → compare to dates[:i]
            if i > 0:
                dprior = cur - dates[:i]
                m7[i]  = np.sum(dprior <= 7)
                m14[i] = np.sum(dprior <= 14)
                m28[i] = np.sum(dprior <= 28)
        PM.loc[idx, "matches_last_7"]  = m7
        PM.loc[idx, "matches_last_14"] = m14
        PM.loc[idx, "matches_last_28"] = m28

        # Rolling workload in minutes (prior only, by match count not days)
        mins = sub["minutes"].fillna(0.0)
        PM.loc[idx, "minutes_last_7"]  = mins.shift(1).rolling(7,  min_periods=1).sum().values
        PM.loc[idx, "minutes_last_14"] = mins.shift(1).rolling(14, min_periods=1).sum().values
        PM.loc[idx, "minutes_last_28"] = mins.shift(1).rolling(28, min_periods=1).sum().values

        # Avg minutes per set over last 10 prior matches (coarse set estimate by best_of)
        best_of = sub["best_of"].fillna(3)
        sets_est = np.where(best_of >= 5, 4.0, 2.5)
        mins_per_set = np.divide(mins, np.maximum(sets_est, 1.0))
        PM.loc[idx, "avg_minutes_per_set_last_10"] = (
            pd.Series(mins_per_set).shift(1).rolling(10, min_periods=3).mean().values
        )

    # back-to-back flag
    PM["back_to_back_flag"] = (PM["days_since_prev"] <= 1).astype(int)

    # ✅ CRITICAL FIX: use transform (keeps original index) instead of apply
    PM["age_at_match"] = (
        PM.groupby("player_id")["age_at_match"]
          .transform(lambda s: s.ffill().bfill())
    )

    # surface one-hot
    PM["surface"] = PM["surface"].fillna("Other")
    for s in ["Hard", "Clay", "Grass", "Carpet", "Other"]:
        PM[f"surf_{s}"] = (PM["surface"] == s).astype(int)

    return PM


def maybe_merge_tfc(PM: pd.DataFrame, tfc_match_path: str | None) -> pd.DataFrame:
    if not tfc_match_path:
        PM["TFC_adj_prev"] = np.nan
        return PM
    try:
        tfc = pd.read_csv(tfc_match_path)
        # unify keys
        # TFC file has: player_id, tourney_id, tourney_date, TFC, TFC_adj ...
        if "tourney_date" in tfc.columns and not np.issubdtype(tfc["tourney_date"].dtype, np.datetime64):
            tfc["tourney_date"] = pd.to_datetime(tfc["tourney_date"], errors="coerce")
        tfc = tfc.rename(columns={"tourney_date": "date"})
        tfc = tfc[["player_id","date","TFC_adj"]].copy()
        tfc["player_id"] = tfc["player_id"].astype(str)

        # prior TFC_adj (shift within player by date)
        tfc = tfc.sort_values(["player_id","date"])
        tfc["TFC_adj_prev"] = tfc.groupby("player_id")["TFC_adj"].shift(1)

        PM = PM.merge(tfc[["player_id","date","TFC_adj_prev"]], on=["player_id","date"], how="left")
        return PM
    except Exception:
        PM["TFC_adj_prev"] = np.nan
        return PM

# ---------------------- Modeling ----------------------

def zscore(df, cols):
    out = df.copy()
    for c in cols:
        mu = out[c].mean(skipna=True)
        sd = out[c].std(skipna=True)
        if not np.isfinite(sd) or sd < 1e-9:
            out[c + "_z"] = 0.0
        else:
            out[c + "_z"] = (out[c] - mu) / sd
    return out

def heuristic_proba(df):
    # lightweight sigmoid over selected z-features
    cols = ["days_since_prev_z", "matches_last_7_z", "minutes_last_14_z",
            "avg_minutes_per_set_last_10_z", "back_to_back_flag", "long_match_flag",
            "age_at_match_z", "TFC_adj_prev_z"]
    w = {
        "days_since_prev_z": -0.6,           # more rest reduces risk
        "matches_last_7_z":  0.7,            # more matches ↑ risk
        "minutes_last_14_z": 0.5,            # more minutes ↑ risk
        "avg_minutes_per_set_last_10_z": 0.4,# high intensity ↑ risk
        "back_to_back_flag": 0.6,            # ↑ risk
        "long_match_flag":  0.5,             # ↑ risk
        "age_at_match_z":   0.3,             # older ↑ risk
        "TFC_adj_prev_z":   0.3,             # travel fatigue ↑ risk
    }
    x = np.zeros(len(df))
    for c in cols:
        if c in df.columns:
            x += w.get(c, 0.0) * df[c].fillna(0.0).values
    # logistic
    proba = 1.0 / (1.0 + np.exp(-x))
    return proba

def fit_predict_rf(df, feature_cols, label_col="injury_label"):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except Exception:
        return None

    # Train on <=2018, apply to all (simple temporal split)
    df_train = df[df["iso_year"] <= 2018].copy()
    df_pred  = df.copy()

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values.astype(int)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42))
    ])
    if len(np.unique(y_train)) < 2:
        return None

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(df_pred[feature_cols].values)[:,1]
    return proba

# ---------------------- Aggregation ----------------------

def aggregate_weekly(PM: pd.DataFrame, by_surface: bool=False) -> pd.DataFrame:
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")
    g = PM.groupby(cols, observed=True)
    out = g.agg(matches=("injury_label","size"),
                irp_mean=("irp_proba","mean")).reset_index()
    return out.sort_values(cols).reset_index(drop=True)

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Injury Risk Predictor (IRP): match/weekly/weekly_surface")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--tfc_match", default=None, help="Optional path to tfc_match.csv to include TFC_adj_prev")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match table…")
    PM = build_player_matches(master)

    print("[INFO] Adding temporal workload features…")
    PM = finalize_workload_features(PM)

    print("[INFO] Merging optional TFC (if provided)…")
    PM = maybe_merge_tfc(PM, args.tfc_match)

    # z-scores for numeric features
    num_cols = ["days_since_prev","matches_last_7","matches_last_14","matches_last_28",
                "minutes_last_7","minutes_last_14","minutes_last_28",
                "avg_minutes_per_set_last_10","age_at_match","TFC_adj_prev"]
    PM = zscore(PM, num_cols)

    # modeling features
    feat = [
        "days_since_prev_z","matches_last_7_z","matches_last_14_z","matches_last_28_z",
        "minutes_last_7_z","minutes_last_14_z","minutes_last_28_z",
        "avg_minutes_per_set_last_10_z","age_at_match_z","TFC_adj_prev_z",
        "back_to_back_flag","long_match_flag",
        "surf_Hard","surf_Clay","surf_Grass","surf_Carpet","surf_Other"
    ]

    print("[INFO] Fitting RandomForest (if sklearn available)…")
    proba_rf = fit_predict_rf(PM, feat, label_col="injury_label")

    if proba_rf is None:
        print("[WARN] sklearn not available or insufficient labels → using heuristic.")
        PM["irp_proba"] = heuristic_proba(PM)
    else:
        PM["irp_proba"] = proba_rf

    # --- Save match-level
    match_cols = ["player_id","player_name","date","iso_year","week_num","week_start","surface",
                  "days_since_prev","matches_last_7","matches_last_14","matches_last_28",
                  "minutes_last_7","minutes_last_14","minutes_last_28",
                  "avg_minutes_per_set_last_10","back_to_back_flag","long_match_flag",
                  "age_at_match","TFC_adj_prev","injury_label","irp_proba"]
    match_out = out_root / "irp_match.csv"
    PM[match_cols].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level IRP → {match_out} ({len(PM):,} rows)")

    # --- Weekly (global)
    weekly = aggregate_weekly(PM, by_surface=False)
    weekly_out = out_root / "irp_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly IRP → {weekly_out} ({len(weekly):,} rows)")

    # --- Weekly (surface)
    weekly_surf = aggregate_weekly(PM, by_surface=True)
    weekly_surf_out = out_root / "irp_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface IRP → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    # quick coverage
    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")

if __name__ == "__main__":
    main()
