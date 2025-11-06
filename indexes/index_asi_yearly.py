# -*- coding: utf-8 -*-
"""
Adaptive Surface Intelligence (ASI)
-----------------------------------
Quantifies a player's long-term surface learning and convergence.
Inputs: surface-level weekly performance (e.g., from SRI/REI/TVC indexes)
Outputs:
  - asi_surface_yearly.csv
  - asi_player_yearly.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Helpers
# ---------------------------
def safe_slope(x, y):
    """Compute slope dy/dx using linear regression with NaN handling."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    X = x[mask] - np.mean(x[mask])
    Y = y[mask] - np.mean(y[mask])
    return (X * Y).sum() / (X**2).sum()


# ---------------------------
# Compute ASI components
# ---------------------------
def compute_asi(df, perf_col="win_rate", w1=0.5, w2=0.3, w3=0.2):
    out = []

    for pid, g in df.groupby("player_id", sort=False):
        for t, gt in g.groupby("iso_year"):
            if gt["surface"].nunique() < 2:
                continue

            # Surface-specific performance values
            surf_perf = gt.groupby("surface")[perf_col].mean()
            p_mean = surf_perf.mean()
            sci = 1 - (surf_perf.std() / (p_mean + 1e-6))  # Surface Convergence Index

            # Compute learning slopes per surface (min 3 years of history)
            prev = g[g["iso_year"] < t]
            slr_list = []
            for s, ss in g.groupby("surface"):
                if ss["iso_year"].nunique() >= 3:
                    slope = safe_slope(ss["iso_year"].values, ss[perf_col].values)
                    slr_list.append(slope)
            slr = np.nanmean(slr_list) if slr_list else np.nan

            # Directional Intelligence: weakest surface improvement
            year_min_surf = surf_perf.idxmin()
            prev_perf = prev[prev["surface"] == year_min_surf][perf_col].mean()
            year_max_surf = surf_perf.idxmax()
            prev_gap = (
                prev[prev["surface"] == year_max_surf][perf_col].mean()
                - prev_perf + 1e-6
            )
            di = (surf_perf[year_min_surf] - prev_perf) / prev_gap if prev_gap > 0 else np.nan

            asi = w1 * sci + w2 * (slr if not np.isnan(slr) else 0) + w3 * (di if not np.isnan(di) else 0)
            out.append({
                "player_id": pid,
                "player_name": gt["player_name"].iloc[0],
                "iso_year": t,
                "SCI": sci,
                "SLR": slr,
                "DI": di,
                "ASI": asi,
            })

    return pd.DataFrame(out)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Adaptive Surface Intelligence (ASI)")
    ap.add_argument("--surface_weekly", required=True, help="Path to a surface-level weekly index (e.g. sri_weekly_surface.csv)")
    ap.add_argument("--out_root", required=True, help="Output folder for ASI CSVs")
    ap.add_argument("--perf_col", default=None, help="Performance metric column (auto-detected if not provided)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading weekly surface performance → {args.surface_weekly}")
    df = pd.read_csv(args.surface_weekly)
    df["iso_year"] = df["iso_year"].astype(int)

    # --- auto-detect performance column ---
    if args.perf_col is not None and args.perf_col in df.columns:
        perf_col = args.perf_col
    else:
        candidates = [c for c in df.columns if any(x in c.lower() for x in ["_mean", "score", "index", "rate"])]
        perf_col = candidates[0] if candidates else None
        if perf_col is None:
            raise ValueError("❌ Could not detect performance column automatically. Please specify with --perf_col")

    print(f"[INFO] Using performance column: {perf_col}")

    # ensure numeric
    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")

    # Aggregate to yearly averages per surface
    yearly = (
        df.groupby(["player_id", "player_name", "iso_year", "surface"], observed=True)
        .agg(**{perf_col: (perf_col, "mean")})
        .reset_index()
        .sort_values(["player_id", "iso_year", "surface"])
    )

    print("[INFO] Computing ASI per player-year…")
    asi_df = compute_asi(yearly, perf_col)

    out1 = out_root / "asi_player_yearly.csv"
    asi_df.to_csv(out1, index=False)
    print(f"[INFO] Saved ASI player-year summary → {out1} ({len(asi_df):,} rows)")

    # Optional: merge back to surfaces
    surf_df = yearly.merge(asi_df, on=["player_id", "player_name", "iso_year"], how="left")
    out2 = out_root / "asi_surface_yearly.csv"
    surf_df.to_csv(out2, index=False)
    print(f"[INFO] Saved surface-level ASI details → {out2} ({len(surf_df):,} rows)")

    if not asi_df.empty:
        print(f"[INFO] Coverage: {int(asi_df['iso_year'].min())}-{int(asi_df['iso_year'].max())} | Players: {asi_df['player_id'].nunique()}")


if __name__ == "__main__":
    main()
