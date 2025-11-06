"""
Tactical Versatility Score (TVS) — Weekly Series (CSV)
=====================================================

Goal
----
Quantify how versatile a player is tactically across time:
- Learn global "style" clusters from per-match tactical features.
- For each player-week, compute the distribution of recently used styles (entropy),
  and the recent win rate. Combine into a Tactical Versatility Score (TVS).

TVS (default):
    TVS = 0.5 * Entropy_norm  +  0.5 * RecentWinRate

Entropy_norm = -Σ p_k log(p_k) / log(K)   in [0, 1], K = number of clusters

Outputs
-------
1) tvs_match_features.csv   : per-player-match features + style cluster
2) tvs_style_centers.csv    : learned style cluster centers in feature space
3) tvs_weekly.csv           : weekly series per player with TVS, entropy, win rate

Run
---
python index_tvs_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" --out_root "D:\\Tennis\\data\\indexes" --n_clusters 6 --window_matches 15
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Utilities and constants
# -----------------------------
TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]  # dataset may include Carpet historically

FEATURE_COLUMNS = [
    # Serving (player's own serve)
    "ace_rate",            # aces / serve points
    "df_rate",             # double faults / serve points
    "fs_in",               # first serve in %
    "fs_win",              # first serve points won %
    "ss_win",              # second serve points won %
    "hold_rate",           # holds ≈ 1 - breaks_conceded / service_games (bounded [0,1])

    # Returning (vs opponent's serve)
    "ret_pts_won",         # return points won % = 1 - (opp_1stWon+opp_2ndWon)/opp_svpt
    "break_rate",          # breaks per return game ≈ (opp_bpFaced - opp_bpSaved) / opp_SvGms (bounded)

    # Context (optionally informative)
    "tier_w",              # tournament tier weight
    "best_of",             # 3 or 5
    "minutes_norm",        # minutes / 180 (cap)
] + [f"surf_{s}" for s in SURFACES]  # one-hot surface


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


def _to_week(dt_series: pd.Series) -> pd.Series:
    """Week starts Monday; returns Timestamp at week start."""
    return dt_series.dt.to_period("W-MON").dt.start_time


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    return np.clip(out, 0.0, 1.0)


# -----------------------------
# Build player-match feature rows
# -----------------------------
def build_player_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From match-level records, produce per-player rows with tactical features.

    Uses:
    - winner-side stats for the winner row (w_*),
    - loser-side stats for the loser row (l_*),
    - return stats inferred from opponent's serve stats.
    """

    # robust date parsing and filtering
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # normalize tournament level
    lvl = df.get("tourney_level_norm", df.get("tourney_level", ""))
    df["tier_norm"] = pd.Series(lvl, index=df.index).astype(str).map(_normalize_tourney_level)
    df["tier_w"] = df["tier_norm"].map(TIER_WEIGHT).fillna(1.0)

    # cap minutes to 180 for normalization (Bo5 marathons), avoid NaN
    df["minutes"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0).clip(lower=0, upper=600)
    df["minutes_norm"] = (df["minutes"] / 180.0).clip(0, 1)

    # helper to build one side
    def side_rows(prefix_player: str, prefix_opp: str, label_val: int):
        # player serve stats
        svpt = pd.to_numeric(df[f"{prefix_player}_svpt"], errors="coerce").fillna(0)
        fs_in = _safe_div(pd.to_numeric(df[f"{prefix_player}_1stIn"], errors="coerce").fillna(0), svpt)
        fs_won = _safe_div(pd.to_numeric(df[f"{prefix_player}_1stWon"], errors="coerce").fillna(0),
                           pd.to_numeric(df[f"{prefix_player}_1stIn"], errors="coerce").fillna(0))
        ss_won = _safe_div(pd.to_numeric(df[f"{prefix_player}_2ndWon"], errors="coerce").fillna(0),
                           (svpt - pd.to_numeric(df[f"{prefix_player}_1stIn"], errors="coerce").fillna(0)))
        ace_rate = _safe_div(pd.to_numeric(df.get(f"{prefix_player}_ace"), errors="coerce").fillna(0), svpt)
        df_rate = _safe_div(pd.to_numeric(df.get(f"{prefix_player}_df"), errors="coerce").fillna(0), svpt)

        sv_gms = pd.to_numeric(df.get(f"{prefix_player}_SvGms"), errors="coerce").fillna(0)
        bp_faced = pd.to_numeric(df.get(f"{prefix_player}_bpFaced"), errors="coerce").fillna(0)
        bp_saved = pd.to_numeric(df.get(f"{prefix_player}_bpSaved"), errors="coerce").fillna(0)
        breaks_conceded = (bp_faced - bp_saved).clip(lower=0)
        hold_rate = 1.0 - _safe_div(breaks_conceded, sv_gms)

        # opponent serve (for player's return)
        opp_svpt = pd.to_numeric(df.get(f"{prefix_opp}_svpt"), errors="coerce").fillna(0)
        opp_1stWon = pd.to_numeric(df.get(f"{prefix_opp}_1stWon"), errors="coerce").fillna(0)
        opp_2ndWon = pd.to_numeric(df.get(f"{prefix_opp}_2ndWon"), errors="coerce").fillna(0)
        ret_pts_won = _safe_div((opp_svpt - opp_1stWon - opp_2ndWon), opp_svpt)

        opp_SvGms = pd.to_numeric(df.get(f"{prefix_opp}_SvGms"), errors="coerce").fillna(0)
        opp_bpFaced = pd.to_numeric(df.get(f"{prefix_opp}_bpFaced"), errors="coerce").fillna(0)
        opp_bpSaved = pd.to_numeric(df.get(f"{prefix_opp}_bpSaved"), errors="coerce").fillna(0)
        breaks_achieved = (opp_bpFaced - opp_bpSaved).clip(lower=0)
        break_rate = _safe_div(breaks_achieved, opp_SvGms)

        # one-hot surface
        surface = df["surface"].fillna("Hard").astype(str)
        surf_oh = {f"surf_{s}": (surface == s).astype(int) for s in SURFACES}
        # best_of
        best_of = pd.to_numeric(df.get("best_of", 3), errors="coerce").fillna(3).clip(lower=1, upper=5)

        rows = pd.DataFrame({
            "player_id": df["winner_id"] if prefix_player == "w" else df["loser_id"],
            "player_name": df["winner_name"] if prefix_player == "w" else df["loser_name"],
            "opp_id": df["loser_id"] if prefix_player == "w" else df["winner_id"],
            "opp_name": df["loser_name"] if prefix_player == "w" else df["winner_name"],
            "date": df["tourney_date"],
            "week": _to_week(df["tourney_date"]),
            "surface": surface,
            "tier_w": df["tier_w"],
            "best_of": best_of,
            "minutes_norm": df["minutes_norm"],
            "label": label_val,

            "ace_rate": ace_rate,
            "df_rate": df_rate,
            "fs_in": fs_in,
            "fs_win": fs_won,
            "ss_win": ss_won,
            "hold_rate": hold_rate,

            "ret_pts_won": ret_pts_won,
            "break_rate": break_rate,
        })
        # attach one-hot surface columns
        for k, v in surf_oh.items():
            rows[k] = v.values
        return rows

    W = side_rows("w", "l", 1)
    L = side_rows("l", "w", 0)
    out = pd.concat([W, L], ignore_index=True).dropna(subset=["player_id", "date"])
    # Keep features bounded
    for col in ["ace_rate", "df_rate", "fs_in", "fs_win", "ss_win", "hold_rate", "ret_pts_won", "break_rate"]:
        out[col] = out[col].astype(float).clip(0.0, 1.0)
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


# -----------------------------
# Learn global style clusters
# -----------------------------
def learn_style_clusters(df_feat: pd.DataFrame, n_clusters: int = 6, random_state: int = 42):
    """Fit a global MiniBatchKMeans on standardized features and assign clusters."""
    X = df_feat[FEATURE_COLUMNS].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048, n_init="auto")
    labels = km.fit_predict(Xs)

    centers = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLUMNS)
    # de-standardize centers back to feature space for interpretability
    centers = pd.DataFrame(scaler.inverse_transform(centers.values), columns=FEATURE_COLUMNS)

    return labels, centers, scaler, km


# -----------------------------
# Compute weekly TVS per player
# -----------------------------
def _entropy_norm(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    Hmax = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(H / Hmax)


def compute_weekly_tvs(df_with_clusters: pd.DataFrame, window_matches: int = 15, n_clusters: int = 6,
                       alpha: float = 0.5, beta: float = 0.5) -> pd.DataFrame:
    """
    For each player-week, look back 'window_matches' matches:
    - style distribution (entropy_norm)
    - recent win rate
    - TVS = alpha * entropy_norm + beta * recent_win_rate
    """
    rows = []
    # ensure chronological per player
    df_with_clusters = df_with_clusters.sort_values(["player_id", "date"])

    for pid, dfp in df_with_clusters.groupby("player_id"):
        # rolling window by match count: we will compute at each week boundary
        # get unique weeks (chronological)
        weeks = dfp["week"].dropna().sort_values().unique()
        # index for quick mask
        for wk in weeks:
            upto = dfp[dfp["date"] <= wk + pd.Timedelta(days=6)]  # include matches in the week
            # last N matches before (and including) this week
            hist = upto.tail(window_matches)
            if hist.empty:
                continue

            # style histogram over last N matches
            counts = pd.Series(0, index=range(n_clusters), dtype=float)
            vc = hist["style_cluster"].value_counts()
            for k, v in vc.items():
                if k in counts.index:
                    counts.loc[int(k)] = float(v)

            ent = _entropy_norm(counts)

            wr = hist["label"].mean() if "label" in hist else 0.0
            tvs = alpha * ent + beta * wr

            rows.append({
                "player_id": pid,
                "player_name": hist["player_name"].iloc[-1],
                "week": pd.to_datetime(wk),
                "matches_window": int(len(hist)),
                "entropy_norm": float(ent),
                "recent_win_rate": float(wr),
                "TVS": float(tvs),
            })

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute Tactical Versatility Score (TVS) Weekly (CSV)")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--n_clusters", type=int, default=6, help="Number of global style clusters")
    ap.add_argument("--window_matches", type=int, default=15, help="Matches lookback window for weekly TVS")
    ap.add_argument("--alpha", type=float, default=0.5, help="Weight for entropy in TVS")
    ap.add_argument("--beta", type=float, default=0.5, help="Weight for recent win rate in TVS")
    ap.add_argument("--random_state", type=int, default=42, help="Random state for clustering")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master from {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building per-player-match features...")
    pm = build_player_match_features(master)
    print(f"[INFO] Built features for {len(pm):,} player-matches")

    # Ensure all feature columns exist (surface one-hots)
    for col in [f"surf_{s}" for s in SURFACES]:
        if col not in pm.columns:
            pm[col] = 0

    # Keep only rows with complete feature set
    missing = [c for c in FEATURE_COLUMNS if c not in pm.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing}")

    print("[INFO] Learning global style clusters...")
    labels, centers, scaler, km = learn_style_clusters(pm[FEATURE_COLUMNS], n_clusters=args.n_clusters,
                                                       random_state=args.random_state)
    pm["style_cluster"] = labels.astype(int)

    # Save per-match assignment
    match_out = out_root / "tvs_match_features.csv"
    cols_out = (["player_id", "player_name", "opp_id", "opp_name", "date", "week", "surface",
                 "style_cluster", "label"] + FEATURE_COLUMNS)
    pm[cols_out].to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level features + clusters → {match_out}")

    # Save style centers in feature space
    centers_out = out_root / "tvs_style_centers.csv"
    centers.to_csv(centers_out, index=False)
    print(f"[INFO] Saved style cluster centers → {centers_out}")

    # Weekly TVS
    print("[INFO] Computing weekly TVS...")
    tvs_weekly = compute_weekly_tvs(pm, window_matches=args.window_matches,
                                    n_clusters=args.n_clusters, alpha=args.alpha, beta=args.beta)
    weekly_out = out_root / "tvs_weekly.csv"
    tvs_weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly TVS → {weekly_out} ({len(tvs_weekly):,} rows)")

    # Small summary
    if not tvs_weekly.empty:
        tail_years = (tvs_weekly["week"].dt.year).dropna()
        print(f"[INFO] Weeks coverage: {tail_years.min()}–{tail_years.max()}, players: {tvs_weekly['player_id'].nunique()}")


if __name__ == "__main__":
    main()
