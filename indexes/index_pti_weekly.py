"""
Peak Timing Index (PTI) — Seasonal profile + Year-Week series
=============================================================

Outputs
-------
1) pti_weekly.csv          : seasonal curve (weeks 1..52 aggregated across all years)
2) pti_weekly_by_year.csv  : full Year + Week time series (1991..2025)
3) pti_summary.csv         : peak week, intensity, span, variability, archetype

Run
---
python index_pti_weekly.py --master "D:\\Tennis\\data\\master\\tennis_master_1991.parquet" ^
                           --out_root "D:\\Tennis\\data\\indexes" ^
                           --smooth_window 5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TIER_WEIGHT = {"G": 1.5, "M": 1.3, "A": 1.1, "B": 1.0}
WEEK_MIN, WEEK_MAX = 1, 52

# ---------- helpers ----------
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

def _iso_year(dt: pd.Series) -> pd.Series:
    # ISO calendar; aligns with week numbering
    return dt.dt.isocalendar().year.astype(int)

def _week_of_year(dt: pd.Series) -> pd.Series:
    w = dt.dt.isocalendar().week.astype(int)
    return w.clip(lower=WEEK_MIN, upper=WEEK_MAX)

def _circular_ma(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    if window % 2 == 0:
        window += 1
    k = window // 2
    ext = np.concatenate([values[-k:], values, values[:k]])
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ext, kernel, mode="valid")

def _tier_weight(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_normalize_tourney_level).map(TIER_WEIGHT).fillna(1.0)

# ---------- build player-match ----------
def build_player_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    tier_raw = df.get("tourney_level_norm", df.get("tourney_level", ""))
    tier_w = _tier_weight(pd.Series(tier_raw, index=df.index))

    date = df["tourney_date"]
    iso_year = _iso_year(date)
    week_num = _week_of_year(date)

    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "label": 1,
        "date": date,
        "iso_year": iso_year,
        "week_num": week_num,
        "tier_w": tier_w,
    })
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "label": 0,
        "date": date,
        "iso_year": iso_year,
        "week_num": week_num,
        "tier_w": tier_w,
    })
    out = pd.concat([W, L], ignore_index=True)
    return out.dropna(subset=["player_id", "date"]).sort_values(["player_id", "date"]).reset_index(drop=True)

# ---------- seasonal curve (across all years; weeks 1..52) ----------
def seasonal_curve(df_pm: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    grp = (
        df_pm.groupby(["player_id", "player_name", "week_num"], observed=True)
        .agg(weighted_wins=("label", lambda s: float((df_pm.loc[s.index, "tier_w"].values * s.values).sum())),
             weighted_total=("tier_w", "sum"),
             matches=("label", "size"))
        .reset_index()
    )
    grp["weighted_total"] = grp["weighted_total"].replace(0, np.nan)
    grp["winrate_woy"] = (grp["weighted_wins"] / grp["weighted_total"]).fillna(0.0).clip(0.0, 1.0)

    def _ensure_weeks(g):
        allw = pd.DataFrame({"week_num": np.arange(WEEK_MIN, WEEK_MAX + 1, dtype=int)})
        g2 = allw.merge(g, on="week_num", how="left")
        g2["player_id"] = g["player_id"].iloc[0]
        g2["player_name"] = g["player_name"].iloc[0]
        for c in ["matches", "weighted_wins", "weighted_total", "winrate_woy"]:
            g2[c] = g2[c].fillna(0.0)
        return g2

    full = grp.groupby(["player_id", "player_name"], group_keys=False).apply(_ensure_weeks).reset_index(drop=True)

    def _smooth_player(g):
        wr = g["winrate_woy"].to_numpy(float)
        sm = _circular_ma(wr, smooth_window)
        g["winrate_smooth"] = sm
        mean_val = float(np.mean(sm)) if np.mean(sm) > 0 else 1.0
        g["PTI"] = (sm / mean_val).astype(float)
        return g

    return full.groupby(["player_id", "player_name"], group_keys=False).apply(_smooth_player).reset_index(drop=True)

# ---------- weekly by YEAR (Year + Week) ----------
def weekly_by_year(df_pm: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    # aggregate tier-weighted winrate per player-year-week
    grp = (
        df_pm.groupby(["player_id", "player_name", "iso_year", "week_num"], observed=True)
        .agg(weighted_wins=("label", lambda s: float((df_pm.loc[s.index, "tier_w"].values * s.values).sum())),
             weighted_total=("tier_w", "sum"),
             matches=("label", "size"))
        .reset_index()
    )
    grp["weighted_total"] = grp["weighted_total"].replace(0, np.nan)
    grp["winrate_week"] = (grp["weighted_wins"] / grp["weighted_total"]).fillna(0.0).clip(0, 1)

    # ensure weeks 1..52 exist in each player-year
    def _ensure_weeks_year(g):
        allw = pd.DataFrame({"week_num": np.arange(WEEK_MIN, WEEK_MAX + 1, dtype=int)})
        g2 = allw.merge(g, on="week_num", how="left")
        g2["player_id"] = g["player_id"].iloc[0]
        g2["player_name"] = g["player_name"].iloc[0]
        g2["iso_year"] = g["iso_year"].iloc[0]
        for c in ["matches", "weighted_wins", "weighted_total", "winrate_week"]:
            g2[c] = g2[c].fillna(0.0)
        return g2

    full = (
        grp.groupby(["player_id", "player_name", "iso_year"], group_keys=False)
        .apply(_ensure_weeks_year)
        .reset_index(drop=True)
    )

    # compute player's global seasonal mean for PTI_global normalization
    # (use seasonal curve across all years)
    seasonal = seasonal_curve(df_pm, smooth_window=smooth_window)
    global_mean = (
        seasonal.groupby("player_id", as_index=False)["winrate_smooth"].mean()
        .rename(columns={"winrate_smooth": "global_mean_wr"})
    )

    # smooth within each player-year and normalize
    def _smooth_year(g):
        wr = g["winrate_week"].to_numpy(float)
        sm = _circular_ma(wr, smooth_window)
        g["winrate_smooth"] = sm
        year_mean = float(np.mean(sm)) if np.mean(sm) > 0 else 1.0
        g["PTI_year"] = (sm / year_mean).astype(float)       # normalized within that year
        return g

    full = (
        full.groupby(["player_id", "player_name", "iso_year"], group_keys=False)
        .apply(_smooth_year)
        .reset_index(drop=True)
    )

    # attach global mean and compute PTI_global
    full = full.merge(global_mean, on="player_id", how="left")
    full["global_mean_wr"] = full["global_mean_wr"].replace(0, np.nan).fillna(1.0)
    full["PTI_global"] = (full["winrate_smooth"] / full["global_mean_wr"]).astype(float)

    # tidy
    cols = ["player_id","player_name","iso_year","week_num","matches",
            "weighted_wins","weighted_total","winrate_week","winrate_smooth",
            "PTI_year","PTI_global"]
    return full[cols].sort_values(["player_id","iso_year","week_num"]).reset_index(drop=True)

# ---------- summary & archetype ----------
def classify_archetype(peak_week: int, pti_var: float, pti_max: float) -> str:
    if pti_var < 0.05:
        return "Consistent performer"
    if 1 <= peak_week <= 15 and pti_max >= 1.10:
        return "Early-season specialist"
    if 16 <= peak_week <= 25 and pti_max >= 1.10:
        return "Clay-season specialist"
    if 24 <= peak_week <= 30 and pti_max >= 1.10:
        return "Grass-season specialist"
    if 35 <= peak_week <= 52 and pti_max >= 1.10:
        return "Late-season closer"
    return "Balanced/unclear"

def seasonal_summary(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, pname), g in curve.groupby(["player_id", "player_name"]):
        pti = g["PTI"].to_numpy(float)
        wks = g["week_num"].to_numpy(int)
        if len(pti):
            pti_max = float(np.nanmax(pti))
            peak_week = int(wks[np.nanargmax(pti)])
            pti_var = float(np.nanvar(pti))
            thr = float(np.nanpercentile(pti, 90.0))
            span = int(np.sum(pti >= thr))
            late_mean = float(g.loc[g["week_num"] >= 35, "PTI"].mean())
        else:
            pti_max = np.nan; peak_week = np.nan; pti_var = np.nan; span = 0; late_mean = np.nan
        arche = classify_archetype(int(peak_week) if not np.isnan(peak_week) else 0, pti_var, pti_max) if not np.isnan(peak_week) else "NA"
        rows.append({
            "player_id": pid, "player_name": pname,
            "PTI_max": pti_max, "peak_week": peak_week,
            "PTI_var": pti_var, "span_top10wks": span,
            "PTI_late_mean": late_mean, "archetype": arche
        })
    return pd.DataFrame(rows).sort_values(["player_name"]).reset_index(drop=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Peak Timing Index (PTI) — seasonal & year-week CSVs")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--smooth_window", type=int, default=5, help="odd window size for circular moving average")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master: {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building player-match records...")
    pm = build_player_match(master)
    print(f"[INFO] Rows: {len(pm):,}  Players: {pm['player_id'].nunique():,}  Years: {pm['iso_year'].min()}–{pm['iso_year'].max()}")

    print(f"[INFO] Seasonal curve (across all years, weeks 1..52) — smooth={args.smooth_window}")
    curve = seasonal_curve(pm, smooth_window=args.smooth_window)
    seasonal_out = out_root / "pti_weekly.csv"
    curve[["player_id","player_name","week_num","matches","weighted_wins","weighted_total","winrate_woy","winrate_smooth","PTI"]].to_csv(seasonal_out, index=False)
    print(f"[INFO] Saved → {seasonal_out} ({len(curve):,} rows)")

    print("[INFO] Year + Week time series (1991..)…")
    by_year = weekly_by_year(pm, smooth_window=args.smooth_window)
    by_year_out = out_root / "pti_weekly_by_year.csv"
    by_year.to_csv(by_year_out, index=False)
    print(f"[INFO] Saved → {by_year_out} ({len(by_year):,} rows)")

    print("[INFO] Summaries + archetypes…")
    summary = seasonal_summary(curve)
    summary_out = out_root / "pti_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"[INFO] Saved → {summary_out} ({len(summary):,} players)")

if __name__ == "__main__":
    main()
