"""
Underdog Success Rate (USR) — Weekly Series (CSV)
=================================================

Goal
----
Quantify how often (and how strongly) a player beats higher-ranked opponents,
adjusted for match difficulty and expected win probability.

Definitions
-----------
underdog_flag  = 1 if player_rank > opp_rank (both ranks present), else 0
rank_diff      = opp_rank - player_rank  (positive => player is underdog)
p_expect       = Logistic model estimate of win prob using rank_diff + tier + surface
upset_flag     = 1 if underdog_flag==1 and label==1 else 0
upset_margin   = (label - p_expect) for underdog matches, else 0

Difficulty weight:
  diff_w = clip( rank_diff / 50, min=0.25, max=3.0 )  # scalable knob

Per-match contributions:
  usr_raw_contrib       = upset_flag
  usr_weighted_contrib  = upset_flag * diff_w * tier_w
  usr_expected_contrib  = upset_margin * tier_w

Weekly aggregation (per player, per week):
  USR_rate              = sum(upset_flag) / max(underdog_matches,1)
  USR_weighted          = sum(usr_weighted_contrib) / max(sum(diff_w*tier_w over underdogs), eps)
  USR_expected_adj      = mean(upset_margin over underdogs)
  Also reports: underdog_matches, upset_wins, matches

Usage
-----
python index_usr_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Constants
# ----------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# ----------------------------
# Helpers
# ----------------------------
def _normalize_tourney_level(x: str) -> str:
    if not isinstance(x, str):
        return "B"
    t = x.strip().upper()
    if "G" in t or "SLAM" in t:
        return "G"
    if "M" in t or "1000" in t or "MAST" in t:
        return "M"
    if "A" in t or "500" in t:
        return "A"
    return "B"

def _to_week(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("W-MON").dt.start_time

def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").fillna(0.0).astype(float)
    b = pd.to_numeric(b, errors="coerce").fillna(0.0).astype(float)
    out = np.zeros_like(a, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return np.clip(out, 0.0, 1.0)

# ----------------------------
# Build player-match table
# ----------------------------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # date
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # tier and weights
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)
    df["tier_w"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    date = df["tourney_date"]
    week = _to_week(date)
    surface = df["surface"].fillna("Hard").astype(str)

    # Winner rows
    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"].astype(str),
        "opp_name": df["loser_name"],
        "player_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "label": 1,
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
    })

    # Loser rows
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"].astype(str),
        "opp_name": df["winner_name"],
        "player_rank": pd.to_numeric(df["loser_rank"], errors="coerce"),
        "opp_rank": pd.to_numeric(df["winner_rank"], errors="coerce"),
        "label": 0,
        "date": date,
        "week": week,
        "surface": surface,
        "tier_w": df["tier_w"],
    })

    out = pd.concat([W, L], ignore_index=True)
    out = out.dropna(subset=["player_id", "opp_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)

    # rank diff and underdog flag
    out["rank_diff"] = (out["opp_rank"] - out["player_rank"]).astype(float)
    out["underdog_flag"] = ((out["player_rank"] > out["opp_rank"]) & (~out["player_rank"].isna()) & (~out["opp_rank"].isna())).astype(int)

    # surface one-hots
    for s in SURFACES:
        out[f"surf_{s}"] = (out["surface"] == s).astype(int)

    return out

# ----------------------------
# Train logistic model: p(win | rank_diff, tier, surface)
# ----------------------------
def fit_logit(df_pm: pd.DataFrame) -> LogisticRegression:
    feats = ["rank_diff", "tier_w"] + [f"surf_{s}" for s in SURFACES]
    X = df_pm[feats].fillna(0.0).astype(float)
    y = df_pm["label"].astype(int)

    # Simple, robust logistic model
    clf = LogisticRegression(solver="liblinear", max_iter=1000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"[INFO] Fitted logistic model on {len(X):,} rows. In-sample acc: {acc:.3f}")
    return clf

# ----------------------------
# Per-match USR metrics
# ----------------------------
def compute_usr_match(df_pm: pd.DataFrame, clf: LogisticRegression) -> pd.DataFrame:
    feats = ["rank_diff", "tier_w"] + [f"surf_{s}" for s in SURFACES]
    X = df_pm[feats].fillna(0.0).astype(float)

    # expected win probability
    p_expect = clf.predict_proba(X)[:, 1]
    df = df_pm.copy()
    df["p_expect"] = p_expect

    # flags and margins
    df["upset_flag"] = ((df["underdog_flag"] == 1) & (df["label"] == 1)).astype(int)
    # Difficulty weight grows with rank_diff (how big the upset would be)
    df["diff_w"] = np.clip(df["rank_diff"] / 50.0, 0.25, 3.0)  # tune scale if desired

    # contributions (only meaningful for underdog matches)
    under_mask = (df["underdog_flag"] == 1)

    df["upset_margin"] = 0.0
    df.loc[under_mask, "upset_margin"] = (df.loc[under_mask, "label"].astype(float) - df.loc[under_mask, "p_expect"])

    df["usr_raw_contrib"] = 0.0
    df.loc[under_mask, "usr_raw_contrib"] = df.loc[under_mask, "upset_flag"].astype(float)

    df["usr_weighted_contrib"] = 0.0
    df.loc[under_mask, "usr_weighted_contrib"] = df.loc[under_mask, "upset_flag"].astype(float) * df.loc[under_mask, "diff_w"] * df.loc[under_mask, "tier_w"]

    df["usr_expected_contrib"] = 0.0
    df.loc[under_mask, "usr_expected_contrib"] = df.loc[under_mask, "upset_margin"] * df.loc[under_mask, "tier_w"]

    return df

# ----------------------------
# Weekly aggregation
# ----------------------------
def aggregate_weekly(df_match: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    # per player-week, we need underdog-only denominators as well as overall matches
    base = df_match.groupby(["player_id", "player_name", "week"], observed=True).agg(
        matches=("label", "size"),
        underdog_matches=("underdog_flag", "sum"),
        upset_wins=("upset_flag", "sum"),
        raw_sum=("usr_raw_contrib", "sum"),
        weighted_sum=("usr_weighted_contrib", "sum"),
        weight_den=("usr_weighted_contrib", lambda s: np.sum((s != 0).astype(float))),  # count of weighted upsets (used only if needed)
        diff_tier_den=("usr_weighted_contrib", lambda s: 1.0),  # placeholder; we'll compute proper denominator below
        expected_mean=("usr_expected_contrib", "mean"),
    ).reset_index()

    # Proper denominator for weighted rate = sum(diff_w * tier_w) over underdog matches that week
    den = (
        df_match[df_match["underdog_flag"] == 1]
        .groupby(["player_id", "player_name", "week"], observed=True)
        .apply(lambda g: float(np.sum(g["diff_w"] * g["tier_w"])) if len(g) else 0.0)
        .reset_index(name="weighted_den")
    )
    out = base.merge(den, on=["player_id", "player_name", "week"], how="left")
    out["weighted_den"] = out["weighted_den"].fillna(0.0)

    # Metrics
    out["USR_rate"] = out["upset_wins"] / np.maximum(out["underdog_matches"], 1)
    out["USR_weighted"] = out["weighted_sum"] / np.maximum(out["weighted_den"], eps)

    # Expectation-adjusted average (only underdogs contribute in match file; mean already respects that)
    out = out.rename(columns={"expected_mean": "USR_expected_adj"})

    # Clean columns
    out = out[[
        "player_id", "player_name", "week",
        "matches", "underdog_matches", "upset_wins",
        "USR_rate", "USR_weighted", "USR_expected_adj"
    ]].sort_values(["player_id", "week"]).reset_index(drop=True)

    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Underdog Success Rate (USR) — Weekly CSV")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match table...")
    pm = build_player_match(master)
    print(f"[INFO] Player-match rows: {len(pm):,}")

    print("[INFO] Fitting logistic expectation model...")
    clf = fit_logit(pm)

    print("[INFO] Computing match-level USR metrics...")
    usr_match = compute_usr_match(pm, clf)

    match_out = out_root / "usr_match.csv"
    cols_out = [
        "player_id","player_name","opp_id","opp_name","date","week","surface","tier_w",
        "player_rank","opp_rank","rank_diff","underdog_flag","label",
        "p_expect","upset_flag","upset_margin","diff_w",
        "usr_raw_contrib","usr_weighted_contrib","usr_expected_contrib"
    ]
    usr_match[cols_out].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level USR → {match_out} ({len(usr_match):,} rows)")

    print("[INFO] Aggregating weekly USR...")
    usr_weekly = aggregate_weekly(usr_match)

    weekly_out = out_root / "usr_weekly.csv"
    usr_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly USR → {weekly_out} ({len(usr_weekly):,} rows)")

    if not usr_weekly.empty:
        years = pd.to_datetime(usr_weekly["week"]).dt.year
        print(f"[INFO] Coverage years: {years.min()}–{years.max()} | Players: {usr_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
