import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TB_PAT = re.compile(r"7[-–]6|6[-–]7", re.IGNORECASE)
BIG_PAT = re.compile(r"(7[-–]5|5[-–]7|7[-–]6|6[-–]7)", re.IGNORECASE)

def to_datetime_yyyymmdd(series):
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

def count_sets(score_str: str) -> int:
    if not isinstance(score_str, str) or not score_str.strip():
        return np.nan
    parts = [p for p in score_str.strip().split() if p.upper() not in {"RET","W/O","WO","DEF","ABN"}]
    return len(parts) if parts else np.nan

def has_tiebreak(score_str: str) -> int:
    if not isinstance(score_str, str): return 0
    return int(bool(TB_PAT.search(score_str)))

def went_to_decider(score_str: str, best_of) -> int:
    nsets = count_sets(score_str)
    if pd.isna(nsets) or pd.isna(best_of): return 0
    max_sets = 5 if int(best_of) >= 5 else 3
    return int(nsets == max_sets)

def comeback_flag(score_str: str) -> int:
    if not isinstance(score_str, str) or not score_str.strip(): return 0
    sets = [s for s in score_str.strip().split() if s.upper() not in {"RET","W/O","WO","DEF","ABN"}]
    if not sets: return 0
    first = sets[0]
    m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)", first)
    if not m: return 0
    w, l = int(m.group(1)), int(m.group(2))
    return int(w < l)

def big_point_density(score_str: str) -> float:
    if not isinstance(score_str, str) or not score_str.strip(): return 0.0
    sets = [p for p in score_str.strip().split() if p.upper() not in {"RET","W/O","WO","DEF","ABN"}]
    if not sets: return 0.0
    bigs = sum(bool(BIG_PAT.search(s)) for s in sets)
    return bigs / len(sets)

def safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return np.divide(a, b)

def z01(x):
    x = pd.Series(x, copy=False)
    lo, hi = x.quantile(0.02), x.quantile(0.98)
    x = x.clip(lo, hi)
    rng = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    return (x - lo) / rng

def canon_surface(s):
    if not isinstance(s, str): return "Other"
    t = s.strip().lower()
    if "hard" in t: return "Hard"
    if "clay" in t: return "Clay"
    if "grass" in t: return "Grass"
    if "carpet" in t: return "Carpet"
    if "indoor" in t: return "Carpet"
    return "Other"

def build_match_evs(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991].copy()
    df["surface_c"] = df["surface"].map(canon_surface)

    df["iso_year"]  = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"]  = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    w_pts_won = df[["w_1stWon","w_2ndWon"]].sum(axis=1, min_count=1)
    l_pts_won = df[["l_1stWon","l_2ndWon"]].sum(axis=1, min_count=1)
    tot_pts   = df[["w_svpt","l_svpt"]].sum(axis=1, min_count=1)

    ratio_w = safe_div(w_pts_won, tot_pts)
    closeness = (1 - (ratio_w - 0.5).abs()*2).clip(lower=0, upper=1).fillna(0.5)

    bp_total = df[["w_bpFaced","l_bpFaced"]].sum(axis=1, min_count=1)
    gms_total = df[["w_SvGms","l_SvGms"]].sum(axis=1, min_count=1).replace(0, np.nan)
    bp_per_game = safe_div(bp_total, gms_total).fillna(0.0)
    pressure = z01(bp_per_game)

    tb = df["score"].map(has_tiebreak).astype(int)
    decider = [went_to_decider(s, b) for s, b in zip(df["score"], df["best_of"])]
    decider = pd.Series(decider, index=df.index, dtype=int)
    comeback = df["score"].map(comeback_flag).astype(int)
    bigpoint = df["score"].map(big_point_density).fillna(0.0)

    dur_density = safe_div(df["minutes"], tot_pts).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    durz = z01(dur_density)

    w_close, w_press, w_tb, w_decider, w_dur, w_cmb, w_big = 0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10

    evs = (
        w_close   * closeness +
        w_press   * pressure +
        w_tb      * tb +
        w_decider * decider +
        w_dur     * durz +
        w_cmb     * comeback +
        w_big     * bigpoint
    ).clip(0, 1)

    out = pd.DataFrame({
        "tourney_id": df["tourney_id"],
        "tourney_name": df.get("tourney_name", ""),
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "surface": df["surface_c"],
        "best_of": df["best_of"],
        "round": df["round"],
        "minutes": df["minutes"],
        "score": df["score"],
        "closeness": closeness,
        "pressure": pressure,
        "tiebreak_flag": tb,
        "decider_flag": decider,
        "comeback_flag": comeback,
        "duration_density": dur_density,
        "big_point_density": bigpoint,
        "EVS": evs
    })
    return out

def expand_to_player_matches(evs_match: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"])
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991].copy()
    df["surface_c"] = df["surface"].map(canon_surface)
    df["iso_year"]  = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"]  = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    W = pd.DataFrame({
        "player_id": df["winner_id"].astype(str),
        "player_name": df["winner_name"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "surface": df["surface_c"],
        "tourney_id": df["tourney_id"],
    })
    L = pd.DataFrame({
        "player_id": df["loser_id"].astype(str),
        "player_name": df["loser_name"],
        "date": df["tourney_date"],
        "iso_year": df["iso_year"],
        "week_num": df["week_num"],
        "week_start": df["week_start"],
        "surface": df["surface_c"],
        "tourney_id": df["tourney_id"],
    })
    PM = pd.concat([W, L], ignore_index=True)
    PM = PM.merge(evs_match[["tourney_id","date","EVS","surface"]]
                  .rename(columns={"surface":"surface_match"}),
                  on=["tourney_id","date"], how="left")
    PM["surface"] = PM["surface"].fillna(PM["surface_match"]).fillna("Other")
    PM = PM.drop(columns=["surface_match"])
    return PM

def aggregate_weekly(PM: pd.DataFrame, by_surface: bool=False) -> pd.DataFrame:
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")
    agg = (
        PM.groupby(cols, observed=True)
          .agg(matches=("EVS","size"), EVS_mean=("EVS","mean"))
          .reset_index()
          .sort_values(cols)
          .reset_index(drop=True)
    )
    return agg

def aggregate_tournament(evs_match: pd.DataFrame) -> pd.DataFrame:
    return (
        evs_match.groupby(["tourney_id","tourney_name","iso_year","surface"], observed=True)
        .agg(matches=("EVS","size"), EVS_mean=("EVS","mean"))
        .reset_index()
        .sort_values(["iso_year","tourney_name"])
        .reset_index(drop=True)
    )

def main():
    ap = argparse.ArgumentParser(description="Entertainment Value Score (EVS) — match / weekly / weekly_surface / tournament")
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Computing match-level EVS with Big Point Density…")
    evs_match = build_match_evs(master)
    evs_match.to_csv(out_root / "evs_match.csv", index=False)

    print("[INFO] Aggregating tournament EVS…")
    tourney = aggregate_tournament(evs_match)
    tourney.to_csv(out_root / "evs_tournament.csv", index=False)
    print(f"[INFO] Saved tournament EVS → {out_root / 'evs_tournament.csv'} ({len(tourney):,} rows)")

    print("[INFO] Expanding to player perspective…")
    PM = expand_to_player_matches(evs_match, master)

    print("[INFO] Aggregating weekly (global)…")
    weekly = aggregate_weekly(PM, by_surface=False)
    weekly.to_csv(out_root / "evs_weekly.csv", index=False)

    print("[INFO] Aggregating weekly (surface)…")
    weekly_surf = aggregate_weekly(PM, by_surface=True)
    weekly_surf.to_csv(out_root / "evs_weekly_surface.csv", index=False)

    print(f"[INFO] Done. Coverage: {weekly['iso_year'].min()}–{weekly['iso_year'].max()} | Players: {weekly['player_id'].nunique()}")

if __name__ == "__main__":
    main()
