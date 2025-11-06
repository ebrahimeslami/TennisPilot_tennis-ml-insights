# -*- coding: utf-8 -*-
"""
Head-to-Head Momentum Index (H2HM)
----------------------------------
Quantifies rivalry momentum between players, weighted by recency,
match importance, and competitiveness.

Outputs:
    - h2hm_match.csv                  (player–opponent per match, directional)
    - h2hm_weekly.csv                 (player weekly mean across opponents)
    - h2hm_weekly_surface.csv         (player weekly mean per surface)
    - h2hm_pairwise_weekly.csv        (pairwise weekly, Δ = H2HM_ij - H2HM_ji)
    - h2hm_pairwise_weekly_surface.csv(pairwise weekly per surface, Δ)

Run:
    python index_h2hm_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" \
      --out_root "D:\\Tennis\\data\\indexes" --window 8
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- helpers ----------
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str):
        return "Other"
    st = s.lower()
    if "hard" in st: return "Hard"
    if "clay" in st: return "Clay"
    if "grass" in st: return "Grass"
    return "Other"

def level_weight(x):
    if not isinstance(x, str): return 0.4
    xl = x.strip().upper()
    if xl.startswith("G"): return 1.0
    if xl.startswith("M"): return 0.85
    if xl.startswith("A") or "500" in xl: return 0.7
    if xl.startswith("B") or "250" in xl: return 0.55
    return 0.45

# ---------- build long ----------
def build_long(master):
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)
    m["imp_w"] = m["tourney_level"].map(level_weight)
    m["iso_year"] = m["tourney_date"].dt.isocalendar().year.astype(int)
    m["week_num"] = m["tourney_date"].dt.isocalendar().week.astype(int)
    m["week_start"] = m["tourney_date"].dt.to_period("W-MON").dt.start_time

    # winner/loser pairs (directional)
    W = pd.DataFrame({
        "player_id": m["winner_id"].astype(str),
        "opp_id": m["loser_id"].astype(str),
        "player_name": m["winner_name"],
        "opp_name": m["loser_name"],
        "surface": m["surface_c"],
        "date": m["tourney_date"],
        "imp_w": m["imp_w"],
        "label": 1,
        "score": m["score"],
        "iso_year": m["iso_year"],
        "week_num": m["week_num"],
        "week_start": m["week_start"]
    })
    L = W.copy()
    L["player_id"], L["opp_id"], L["player_name"], L["opp_name"], L["label"] = \
        m["loser_id"].astype(str), m["winner_id"].astype(str), m["loser_name"], m["winner_name"], 0
    return pd.concat([W, L], ignore_index=True)

# ---------- competitiveness weight ----------
def comp_weight(score):
    if not isinstance(score, str) or "-" not in score:
        return 0.8
    sets = score.strip().split()
    diffs = []
    for s in sets:
        try:
            a, b = map(int, s.split("-"))
            diffs.append(abs(a - b))
        except Exception:
            pass
    if not diffs:
        return 0.8
    avg_diff = np.mean(diffs)
    w = 1 - (avg_diff / 6.0)  # typical set diff range 0–6
    return float(np.clip(w, 0.5, 1.0))

# ---------- H2HM computation (directional) ----------
def compute_h2hm(df_long, N=8, lambda_days=1/180):
    df = df_long.copy()
    df["comp_w"] = df["score"].map(comp_weight)

    def per_pair(g):
        g = g.sort_values("date").copy()
        for idx in range(len(g)):
            sub = g.iloc[max(0, idx - N):idx]
            if sub.empty:
                g.loc[g.index[idx], "H2HM"] = np.nan
                continue
            days = (g.loc[g.index[idx], "date"] - sub["date"]).dt.days.clip(lower=0)
            w_rec = np.exp(-lambda_days * days)
            W = w_rec * sub["imp_w"] * sub["comp_w"]
            h2hm = np.sum(sub["label"] * W) / np.sum(W)
            g.loc[g.index[idx], "H2HM"] = h2hm
        return g

    return (
        df.groupby(["player_id", "opp_id", "surface"], group_keys=False)
          .apply(per_pair)
          .reset_index(drop=True)
    )

# ---------- weekly aggregates (player-centric) ----------
def weekly_aggregate(df, by_surface=False):
    if "week_start" not in df.columns:
        df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    keys = ["player_id", "iso_year", "week_num"]
    if by_surface: keys.append("surface")
    agg = (df.groupby(keys, observed=True)
             .agg(player_name=("player_name","last"),
                  week_start=("week_start","first"),
                  matches=("label","count"),
                  win_rate=("label","mean"),
                  H2HM=("H2HM","mean"))
             .reset_index()
             .sort_values(keys))
    return agg

# ---------- pairwise directional Δ weekly ----------
def pairwise_weekly_delta(df, by_surface=False):
    """
    Build weekly pairwise directional table:
      H2HM_ij (A vs B) and H2HM_ji (B vs A), with Δ = H2HM_ij - H2HM_ji.
    """
    if "week_start" not in df.columns:
        df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time

    keys = ["player_id", "opp_id", "iso_year", "week_num"]
    if by_surface:
        keys.append("surface")

    # Take mean H2HM within the week (directional)
    W = (df.groupby(keys, observed=True)
           .agg(player_name=("player_name","last"),
                opp_name=("opp_name","last"),
                week_start=("week_start","first"),
                H2HM=("H2HM","mean"),
                matches=("label","count"))
           .reset_index())

    # Build reversed copy (swap i/j)
    WR = W.rename(columns={
        "player_id":"opp_id",
        "opp_id":"player_id",
        "player_name":"opp_name",
        "opp_name":"player_name",
        "H2HM":"H2HM_rev",
        "matches":"matches_rev"
    })
    # Merge i→j with j→i on same week (and surface if present)
    on_cols = ["player_id","opp_id","iso_year","week_num"]
    if by_surface:
        on_cols.append("surface")
    P = W.merge(WR[on_cols + ["H2HM_rev","matches_rev"]], on=on_cols, how="left")

    # Canonical pair id to deduplicate (a_id < b_id)
    P["a_id"] = P[["player_id","opp_id"]].min(axis=1)
    P["b_id"] = P[["player_id","opp_id"]].max(axis=1)
    # Keep one orientation (player_id == a_id)
    P = P[P["player_id"] == P["a_id"]].copy()

    # Directional delta from (a→b) perspective
    P["delta"] = P["H2HM"] - P["H2HM_rev"]

    cols = ["a_id","b_id","iso_year","week_num","week_start","H2HM","H2HM_rev","delta","matches","matches_rev"]
    if by_surface:
        cols.insert(2, "surface")
    # Names: keep last seen for each side
    P.rename(columns={
        "player_name":"a_name",
        "opp_name":"b_name"
    }, inplace=True)
    cols_names = ["a_name","b_name"]
    # order final columns
    out_cols = ["a_id","a_name","b_id","b_name"]
    if by_surface:
        out_cols += ["surface"]
    out_cols += ["iso_year","week_num","week_start","H2HM","H2HM_rev","delta","matches","matches_rev"]

    return P[out_cols].reset_index(drop=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lambda_days", type=float, default=1/180)
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    print(f"[INFO] Loading master → {args.master}")
    df_long = build_long(master)

    print("[INFO] Computing H2HM momentum…")
    df_h2hm = compute_h2hm(df_long, N=args.window, lambda_days=args.lambda_days)

    # match-level save
    path_match = f"{args.out_root}/h2hm_match.csv"
    df_h2hm.to_csv(path_match, index=False)
    print(f"[INFO] Saved match-level H2HM → {path_match} ({len(df_h2hm):,} rows)")

    # weekly (player-centric)
    weekly = weekly_aggregate(df_h2hm, by_surface=False)
    weekly.to_csv(f"{args.out_root}/h2hm_weekly.csv", index=False)
    print(f"[INFO] Saved weekly H2HM → {args.out_root}/h2hm_weekly.csv")

    weekly_surf = weekly_aggregate(df_h2hm, by_surface=True)
    weekly_surf.to_csv(f"{args.out_root}/h2hm_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface H2HM → {args.out_root}/h2hm_weekly_surface.csv")

    # pairwise directional Δ (global + surface)
    pair_w = pairwise_weekly_delta(df_h2hm, by_surface=False)
    pair_w.to_csv(f"{args.out_root}/h2hm_pairwise_weekly.csv", index=False)
    print(f"[INFO] Saved pairwise weekly H2HM Δ → {args.out_root}/h2hm_pairwise_weekly.csv")

    pair_ws = pairwise_weekly_delta(df_h2hm, by_surface=True)
    pair_ws.to_csv(f"{args.out_root}/h2hm_pairwise_weekly_surface.csv", index=False)
    print(f"[INFO] Saved pairwise weekly surface H2HM Δ → {args.out_root}/h2hm_pairwise_weekly_surface.csv")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly.iso_year.min())}-{int(weekly.iso_year.max())} | Players: {weekly.player_id.nunique()}")

if __name__ == "__main__":
    main()
