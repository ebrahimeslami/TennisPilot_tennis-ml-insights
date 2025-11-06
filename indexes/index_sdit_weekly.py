# -*- coding: utf-8 -*-
"""
Stamina Depletion Trend Index (SDIT)
------------------------------------
Tracks week-to-week change in a player's stamina (SDI) trajectory.

Inputs:
    - sdi_weekly.csv from SDI computation

Outputs:
    - sdit_weekly.csv
    - sdit_weekly_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- Core Computation ----------
def compute_sdit(df_sdi_weekly, window=8):
    """Compute week-to-week stamina trend index (SDIT)."""
    df = df_sdi_weekly.copy()
    df = df.dropna(subset=["SDI_mean"]).sort_values(["player_id","iso_year","week_num"]).reset_index(drop=True)

    def per_player(g):
        g = g.sort_values(["iso_year","week_num"]).copy()
        # create continuous week index
        g["week_idx"] = np.arange(len(g))
        # compute rolling slope and CV
        sdit_vals = []
        for i in range(len(g)):
            sub = g.iloc[max(0, i - window):i + 1]
            if len(sub) < 3:
                sdit_vals.append(np.nan)
                continue
            x = sub["week_idx"]
            y = sub["SDI_mean"]
            slope = np.cov(x, y, ddof=0)[0,1] / np.var(x)
            cv = (np.std(y) / np.mean(y)) if np.mean(y) > 0 else 0
            sdit_vals.append((slope, cv))
        g["slope"], g["cv"] = zip(*[(s[0], s[1]) if isinstance(s, tuple) else (np.nan, np.nan) for s in sdit_vals])
        return g

    out = df.groupby("player_id", group_keys=False).apply(per_player).reset_index(drop=True)
    # Normalize slopes
    mean_slope = out["slope"].mean(skipna=True)
    std_slope = out["slope"].std(skipna=True)
    out["z_slope"] = ((out["slope"] - mean_slope) / (3 * std_slope)).clip(-1,1)
    out["SDIT"] = (0.7 * out["z_slope"] + 0.3 * (1 - out["cv"].clip(0,1))).clip(-1,1)
    return out

# ---------- Weekly Aggregation ----------
def weekly_surface_aggregate(df, by_surface=False):
    keys = ["player_id","iso_year","week_num"]
    if by_surface and "surface" in df.columns:
        keys.append("surface")
    agg = (
        df.groupby(keys, observed=True)
        .agg(
            player_name=("player_name","last"),
            week_start=("week_start","first"),
            SDI_mean=("SDI_mean","mean"),
            SDIT=("SDIT","mean")
        )
        .reset_index()
        .sort_values(keys)
    )
    return agg

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute Stamina Depletion Trend Index (SDIT)")
    ap.add_argument("--sdi_weekly", required=True, help="Path to sdi_weekly.csv")
    ap.add_argument("--out_root", required=True, help="Output directory")
    ap.add_argument("--window", type=int, default=8, help="Rolling window (weeks)")
    args = ap.parse_args()

    print(f"[INFO] Loading SDI weekly → {args.sdi_weekly}")
    df_sdi = pd.read_csv(args.sdi_weekly)
    if "SDI_mean" not in df_sdi.columns:
        raise ValueError("Input must contain 'SDI_mean' column (from sdi_weekly.csv)")

    print("[INFO] Computing SDIT...")
    df_sdit = compute_sdit(df_sdi, window=args.window)

    # Save outputs
    out_week = f"{args.out_root}/sdit_weekly.csv"
    df_sdit.to_csv(out_week, index=False)
    print(f"[INFO] Saved weekly SDIT → {out_week} ({len(df_sdit):,} rows)")

    # Aggregate by surface if available
    if "surface" in df_sdi.columns:
        df_sdi_surf = df_sdi.copy()
        df_sdit_surf = compute_sdit(df_sdi_surf, window=args.window)
        weekly_surf = weekly_surface_aggregate(df_sdit_surf, by_surface=True)
        out_surf = f"{args.out_root}/sdit_weekly_surface.csv"
        weekly_surf.to_csv(out_surf, index=False)
        print(f"[INFO] Saved weekly surface SDIT → {out_surf}")

    if not df_sdit.empty:
        print(f"[INFO] Coverage: {int(df_sdit['iso_year'].min())}-{int(df_sdit['iso_year'].max())} | Players: {df_sdit['player_id'].nunique()}")

if __name__ == "__main__":
    main()
