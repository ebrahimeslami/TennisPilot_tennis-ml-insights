# -*- coding: utf-8 -*-
"""
Stamina Depletion Index (SDI) + Recovery Curve (RCSDI)
------------------------------------------------------
Integrated script producing stamina depletion and recovery metrics
for all players from match-level data.

Outputs:
    - sdi_match.csv
    - sdi_weekly.csv
    - sdi_weekly_surface.csv
    - sdi_recovery_match.csv
    - sdi_recovery_weekly.csv
    - sdi_recovery_surface_weekly.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------- helpers ----------
def to_datetime_yyyymmdd(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    st = s.lower()
    if "clay" in st: return "Clay"
    if "hard" in st: return "Hard"
    if "grass" in st: return "Grass"
    return "Other"

def surface_friction(surf):
    return {"Clay":1.0, "Hard":0.7, "Grass":0.4}.get(surf, 0.6)

def parse_games(score, win=True):
    """Approximate games won/lost from score."""
    if not isinstance(score, str) or "-" not in score:
        return 0
    total = 0
    for s in score.split():
        try:
            a, b = map(int, s.split("-"))
            total += a if win else b
        except Exception:
            continue
    return total

# ---------- build long ----------
def build_long(master):
    m = master.copy()
    m["tourney_date"] = to_datetime_yyyymmdd(m["tourney_date"])
    m = m[m["tourney_date"].dt.year >= 1991].copy()
    m["surface_c"] = m["surface"].map(canon_surface)

    W = pd.DataFrame({
        "player_id": m["winner_id"].astype(str),
        "player_name": m["winner_name"],
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "minutes": m["minutes"],
        "score": m["score"],
        "label": 1
    })
    L = pd.DataFrame({
        "player_id": m["loser_id"].astype(str),
        "player_name": m["loser_name"],
        "date": m["tourney_date"],
        "surface": m["surface_c"],
        "minutes": m["minutes"],
        "score": m["score"],
        "label": 0
    })
    df = pd.concat([W, L], ignore_index=True)
    df = df.dropna(subset=["date"]).sort_values(["player_id","date"]).reset_index(drop=True)

    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    return df

# ---------- SDI computation ----------
def compute_sdi(df_long):
    df = df_long.copy()
    df["fric"] = df["surface"].map(surface_friction)
    df["games_won"] = df["score"].map(lambda x: parse_games(x, True))
    df["games_lost"] = df["score"].map(lambda x: parse_games(x, False))
    df["total_games"] = df["games_won"] + df["games_lost"]
    df["depletion_raw"] = 1 - (df["games_won"] / df["total_games"].replace(0, np.nan))
    df["depletion_raw"] = df["depletion_raw"].clip(0, 1)

    def per_player(g):
        g = g.sort_values("date").copy()
        g["minutes_prev"] = g["minutes"].shift(1).fillna(0)
        g["sets_prev"] = np.ceil(g["total_games"].shift(1) / 12).fillna(0)
        g["days_since_prev"] = (g["date"] - g["date"].shift(1)).dt.days.fillna(10)

        g["fatigue_load"] = (0.4*(g["minutes_prev"]/180) +
                             0.3*(g["sets_prev"]/5) +
                             0.3*(1/np.maximum(g["days_since_prev"],1))).clip(0,1)

        g["exp_depletion"] = (0.2 + 0.3*np.log1p(g["minutes"]/90) + 0.5*g["fric"]).clip(0,1.5)
        g["SDI"] = (1 - (g["depletion_raw"]/g["exp_depletion"]) * (1 - g["fatigue_load"])).clip(0,1)
        return g

    out = df.groupby("player_id", group_keys=False).apply(per_player)
    cols = ["player_id","player_name","date","iso_year","week_num","week_start",
            "surface","minutes","depletion_raw","fatigue_load","exp_depletion","SDI"]
    return out[cols].reset_index(drop=True)

# ---------- Recovery computation ----------
def compute_recovery(df_sdi: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery delay (days until SDI≥0.7 after fatigue event)."""
    df = df_sdi.copy()  # <— ensures df is defined immediately
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    df["RCSDI"] = np.nan
    df["recovery_days"] = np.nan

    for pid, g in df.groupby("player_id"):
        g = g.sort_values("date").copy()
        for i in range(len(g)):
            fatigue_flag = (g.loc[g.index[i], "fatigue_load"] > 0.6 or g.loc[g.index[i], "minutes"] > 150)
            if fatigue_flag:
                start_date = g.loc[g.index[i], "date"]
                recovery = 30
                for j in range(i+1, min(i+15, len(g))):
                    if g.loc[g.index[j], "SDI"] >= 0.7:
                        recovery = (g.loc[g.index[j], "date"] - start_date).days
                        break
                df.loc[g.index[i], "recovery_days"] = recovery
                df.loc[g.index[i], "RCSDI"] = 1 - min(recovery, 30) / 30
    return df

# ---------- Weekly aggregation ----------
def weekly_aggregate(df, by_surface=False, metric="SDI"):
    keys = ["player_id","iso_year","week_num"]
    if by_surface:
        keys.append("surface")
    metrics = [metric]
    cols = {
        "player_name": ("player_name","last"),
        "week_start": ("week_start","first"),
        "matches": (metric,"count"),
        f"{metric}_mean": (metric,"mean")
    }
    if metric == "SDI":
        cols.update({
            "fatigue_mean": ("fatigue_load","mean"),
            "depletion_mean": ("depletion_raw","mean"),
        })
    if metric == "RCSDI":
        cols.update({
            "fatigue_events": ("RCSDI", lambda x: (x > 0).sum()),
            "recovery_days_mean": ("recovery_days","mean")
        })
    agg = df.groupby(keys, observed=True).agg(**cols).reset_index().sort_values(keys)
    return agg

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    print(f"[INFO] Loading master → {args.master}")

    df_long = build_long(master)
    print("[INFO] Computing base SDI...")
    df_sdi = compute_sdi(df_long)

    # Save base SDI match
    df_sdi.to_csv(f"{args.out_root}/sdi_match.csv", index=False)
    print(f"[INFO] Saved match-level SDI → {args.out_root}/sdi_match.csv")

    # Weekly global
    weekly = weekly_aggregate(df_sdi, by_surface=False, metric="SDI")
    weekly.to_csv(f"{args.out_root}/sdi_weekly.csv", index=False)
    print(f"[INFO] Saved weekly SDI → {args.out_root}/sdi_weekly.csv")

    # Weekly surface
    weekly_surf = weekly_aggregate(df_sdi, by_surface=True, metric="SDI")
    weekly_surf.to_csv(f"{args.out_root}/sdi_weekly_surface.csv", index=False)
    print(f"[INFO] Saved weekly surface SDI → {args.out_root}/sdi_weekly_surface.csv")

    # Compute recovery
    print("[INFO] Computing recovery metrics (RCSDI)...")
    df_rec = compute_recovery(df_sdi)
    df_rec.to_csv(f"{args.out_root}/sdi_recovery_match.csv", index=False)
    print(f"[INFO] Saved match-level recovery metrics → {args.out_root}/sdi_recovery_match.csv")

    weekly_rec = weekly_aggregate(df_rec, by_surface=False, metric="RCSDI")
    weekly_rec.to_csv(f"{args.out_root}/sdi_recovery_weekly.csv", index=False)
    print(f"[INFO] Saved weekly recovery → {args.out_root}/sdi_recovery_weekly.csv")

    weekly_rec_surf = weekly_aggregate(df_rec, by_surface=True, metric="RCSDI")
    weekly_rec_surf.to_csv(f"{args.out_root}/sdi_recovery_surface_weekly.csv", index=False)
    print(f"[INFO] Saved weekly surface recovery → {args.out_root}/sdi_recovery_surface_weekly.csv")

    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly.iso_year.min())}-{int(weekly.iso_year.max())} | Players: {weekly.player_id.nunique()}")

if __name__ == "__main__":
    main()
