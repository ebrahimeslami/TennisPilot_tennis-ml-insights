"""
Travel Fatigue Coefficient (TFC) — Weekly + Weekly Surface + Match
==================================================================
Quantifies travel-induced fatigue based on consecutive tournaments:
- Great-circle distance (km)
- Time zone difference (hours)
- Days of rest between events
- Surface adjustment factor

Per-tournament occurrence TSF:
    TSF = distance_km / (rest_days + 1) + 200 * |tz_diff_hours|

Normalize to [0,1] per season-player:
    TFC = (TSF - min) / (max - min)  (safe if max==min)

Surface adjustment:
    factor = {'Hard':1.0, 'Clay':1.2, 'Grass':0.9, 'Carpet':0.9, 'Indoor':0.8}
    TFC_adj = TFC * factor(surface)

Outputs (CSV in --out_root):
- tfc_match.csv
- tfc_weekly.csv
- tfc_weekly_surface.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- Config -----------------------
SURF_FACTOR = {
    "Hard": 1.0, "Clay": 1.2, "Grass": 0.9, "Carpet": 0.9,
    "Indoor": 0.8, "Unknown": 1.0, None: 1.0
}

# Fallback coordinates/timezones for common tournaments (approx)
# timezone in hours offset from UTC (standard time; DST ignored for simplicity)
BUILTIN_VENUES = {
    # Slam examples
    "Australian Open": ( -37.821, 144.978, 10),   # Melbourne UTC+10
    "Roland Garros":   (  48.847,   2.253,  1),   # Paris UTC+1
    "Wimbledon":       (  51.434,  -0.214,  0),   # London UTC+0
    "US Open":         (  40.749, -73.847, -5),   # NYC UTC-5
    # Masters (selected)
    "Indian Wells":    (  33.723, -116.305, -8),
    "Miami":           (  25.708,  -80.162, -5),
    "Monte Carlo":     (  43.734,    7.421,  1),
    "Madrid":          (  40.407,   -3.690,  1),
    "Rome":            (  41.934,   12.454,  1),
    "Canada":          (  43.634,  -79.420, -5),  # alt Montreal/Toronto
    "Cincinnati":      (  39.346,  -84.301, -5),
    "Shanghai":        (  31.192,  121.336,  8),
    "Paris Masters":   (  48.833,    2.373,  1),
    # Finals-like
    "Tour Finals":     (  51.507,   -0.127,  0),
    # A few others
    "Doha":            (  25.285,   51.531,  3),
    "Dubai":           (  25.226,   55.281,  4),
    "Tokyo":           (  35.644,  139.747,  9),
    "Beijing":         (  39.991,  116.390,  8),
    "Basel":           (  47.553,    7.591,  1),
    "Vienna":          (  48.206,   16.332,  1),
}

# -------------------- Utilities -----------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

def approx_tz_from_lon(lon):
    if pd.isna(lon): return np.nan
    return int(np.round(lon / 15.0))

def normalize01(series):
    smin, smax = series.min(), series.max()
    if pd.isna(smin) or pd.isna(smax) or smax - smin < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)

def to_datetime_yyyymmdd(obj):
    # accepts int or str like 20241229
    return pd.to_datetime(pd.Series(obj).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

# ---------------- Venue lookup -----------------------
def load_venue_map(venue_map_path: str | None) -> pd.DataFrame:
    rows = []
    # built-in dictionary to dataframe
    for name, (lat, lon, tz) in BUILTIN_VENUES.items():
        rows.append({"tourney_name": name, "lat": lat, "lon": lon, "timezone": tz})
    df_builtin = pd.DataFrame(rows)

    if venue_map_path:
        try:
            vm = pd.read_csv(venue_map_path)
            # expected cols: tourney_id,tourney_name,lat,lon,timezone
            # we keep both id and name joins possible
            return vm, df_builtin
        except Exception:
            return None, df_builtin
    return None, df_builtin

def attach_coords(df_t: pd.DataFrame, user_map: pd.DataFrame | None, builtin_map: pd.DataFrame) -> pd.DataFrame:
    out = df_t.copy()

    # First, try user map by id (best), then by name
    if user_map is not None:
        if "tourney_id" in user_map.columns:
            out = out.merge(user_map[["tourney_id","lat","lon","timezone"]].drop_duplicates("tourney_id"),
                            on="tourney_id", how="left", suffixes=("",""))
        if out["lat"].isna().any() and "tourney_name" in user_map.columns:
            out = out.merge(user_map[["tourney_name","lat","lon","timezone"]].drop_duplicates("tourney_name"),
                            on="tourney_name", how="left", suffixes=("",""), copy=False)

    # Fallback: builtin by name (contains common events)
    needs = out["lat"].isna()
    if needs.any():
        out = out.merge(builtin_map, on="tourney_name", how="left", suffixes=("","_b"))
        # fill missing from builtin
        for col in ["lat","lon","timezone"]:
            out[col] = out[col].fillna(out[f"{col}_b"])
        # drop helper
        out = out.drop(columns=[c for c in out.columns if c.endswith("_b")], errors="ignore")

    # Final fallback: if timezone missing but lon present -> approximate
    out["timezone"] = out["timezone"].where(~out["timezone"].isna(), out["lon"].map(approx_tz_from_lon))
    return out

# ---------------- Core build -------------------------
def build_tournament_occurrences(master: pd.DataFrame, venue_map_path: str | None):
    df = master.copy()

    # robust date parse
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"]).values
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    # basic fields
    df["surface"] = df["surface"].fillna("Unknown").str.capitalize()
    # per player, we’ll work at player-tournament occurrence level using winner & loser roles

    # derive player-tournament occurrence (winner)
    W = (df.groupby(["winner_id","winner_name","tourney_id","tourney_name","surface","tourney_date"], as_index=False)
            .size().rename(columns={"winner_id":"player_id","winner_name":"player_name","size":"matches"}))
    L = (df.groupby(["loser_id","loser_name","tourney_id","tourney_name","surface","tourney_date"], as_index=False)
            .size().rename(columns={"loser_id":"player_id","loser_name":"player_name","size":"matches"}))
    PT = pd.concat([W,L], ignore_index=True)
    PT["player_id"] = PT["player_id"].astype(str)

    # attach venue coordinates/timezones
    # attach venue coordinates/timezones
    user_map, builtin_map = load_venue_map(venue_map_path)

    # initialize empty columns so attach_coords() always finds them
    for c in ["lat", "lon", "timezone"]:
        if c not in PT.columns:
            PT[c] = np.nan

    PT = attach_coords(PT, user_map, builtin_map)

    # compute iso week keys
    PT["iso_year"] = PT["tourney_date"].dt.isocalendar().year.astype(int)
    PT["week_num"] = PT["tourney_date"].dt.isocalendar().week.astype(int)
    PT["week_start"] = PT["tourney_date"].dt.to_period("W-MON").dt.start_time

    # sort for transitions
    PT = PT.sort_values(["player_id","tourney_date"]).reset_index(drop=True)
    return PT

def compute_transitions(PT: pd.DataFrame) -> pd.DataFrame:
    """
    For each player's consecutive tournament occurrences, compute:
    distance_km, tz_diff_hours, rest_days, TSF, and normalized TFC per season.
    """
    PT = PT.copy()

    # shift previous occurrence per player
    PT["prev_date"] = PT.groupby("player_id")["tourney_date"].shift(1)
    PT["prev_lat"]  = PT.groupby("player_id")["lat"].shift(1)
    PT["prev_lon"]  = PT.groupby("player_id")["lon"].shift(1)
    PT["prev_tz"]   = PT.groupby("player_id")["timezone"].shift(1)

    # compute metrics
    PT["distance_km"] = [haversine_km(a,b,c,d) for a,b,c,d in zip(PT["prev_lat"],PT["prev_lon"],PT["lat"],PT["lon"])]
    PT["tz_diff"]     = (PT["timezone"] - PT["prev_tz"]).abs()
    PT["rest_days"]   = (PT["tourney_date"] - PT["prev_date"]).dt.days

    # safe defaults
    PT["distance_km"] = PT["distance_km"].fillna(0.0)
    PT["tz_diff"]     = PT["tz_diff"].fillna(0.0)
    PT["rest_days"]   = PT["rest_days"].fillna(14)  # if first event or unknown, assume 2 weeks rest

    # Travel Stress Factor
    PT["TSF"] = PT["distance_km"] / (PT["rest_days"] + 1.0) + 200.0 * PT["tz_diff"]

    # Normalize within player-season to produce TFC
    def _per_season_norm(g):
        g = g.sort_values("tourney_date")
        g["TFC"] = normalize01(g["TSF"])
        return g
    PT = PT.groupby(["player_id","iso_year"], group_keys=False).apply(_per_season_norm)

    # Surface-adjusted TFC
    PT["surf_factor"] = PT["surface"].map(SURF_FACTOR).fillna(1.0)
    PT["TFC_adj"] = PT["TFC"] * PT["surf_factor"]

    return PT

def explode_to_matches(master: pd.DataFrame, PT: pd.DataFrame) -> pd.DataFrame:
    """
    Attach per-occurrence TFC to each match (both winner and loser rows),
    yielding match-level fatigue for each player.
    """
    df = master.copy()
    df["tourney_date"] = to_datetime_yyyymmdd(df["tourney_date"]).values
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]
    df["surface"] = df["surface"].fillna("Unknown").str.capitalize()
    df["iso_year"] = df["tourney_date"].dt.isocalendar().year.astype(int)
    df["week_num"] = df["tourney_date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["tourney_date"].dt.to_period("W-MON").dt.start_time

    # Winner rows
    W = df[[
        "tourney_id","tourney_name","tourney_date","surface","iso_year","week_num","week_start",
        "winner_id","winner_name","loser_id","loser_name","score","minutes"
    ]].copy()
    W.rename(columns={"winner_id":"player_id","winner_name":"player_name",
                      "loser_id":"opp_id","loser_name":"opp_name"}, inplace=True)
    W["label"] = 1

    # Loser rows
    L = df[[
        "tourney_id","tourney_name","tourney_date","surface","iso_year","week_num","week_start",
        "loser_id","loser_name","winner_id","winner_name","score","minutes"
    ]].copy()
    L.rename(columns={"loser_id":"player_id","loser_name":"player_name",
                      "winner_id":"opp_id","winner_name":"opp_name"}, inplace=True)
    L["label"] = 0

    PM = pd.concat([W,L], ignore_index=True)
    PM["player_id"] = PM["player_id"].astype(str)

    # Merge per-occurrence TFC
    occ_cols = ["player_id","tourney_id","tourney_date","TFC","TFC_adj","distance_km","tz_diff","rest_days"]
    PM = PM.merge(PT[occ_cols], on=["player_id","tourney_id","tourney_date"], how="left")

    return PM

def aggregate_weekly(PM: pd.DataFrame, by_surface: bool=False) -> pd.DataFrame:
    cols = ["player_id","player_name","iso_year","week_num","week_start"]
    if by_surface:
        cols.append("surface")

    g = PM.groupby(cols, observed=True)
    out = g.agg(
        matches=("label","size"),
        TFC=("TFC","mean"),
        TFC_adj=("TFC_adj","mean"),
        distance_km=("distance_km","mean"),
        tz_diff=("tz_diff","mean"),
        rest_days=("rest_days","mean"),
    ).reset_index()

    return out.sort_values(cols).reset_index(drop=True)

# ---------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Travel Fatigue Coefficient (TFC) — match/weekly/weekly_surface")
    ap.add_argument("--master", required=True, help="Path to master parquet (1991+)")
    ap.add_argument("--out_root", required=True, help="Output directory for CSVs")
    ap.add_argument("--venue_map", default=None, help="Optional CSV mapping: tourney_id,tourney_name,lat,lon,timezone")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading master → {args.master}")
    master = pd.read_parquet(args.master)

    print("[INFO] Building tournament occurrences per player…")
    PT = build_tournament_occurrences(master, args.venue_map)
    print(f"[INFO] Occurrences: {len(PT):,} rows | Players: {PT['player_id'].nunique()} | Years: {PT['iso_year'].min()}–{PT['iso_year'].max()}")

    print("[INFO] Computing transitions and TFC…")
    PT = compute_transitions(PT)

    print("[INFO] Attaching TFC to match-level rows…")
    PM = explode_to_matches(master, PT)
    match_out = out_root / "tfc_match.csv"
    PM.to_csv(match_out, index=False)
    print(f"[INFO] Saved match-level TFC → {match_out} ({len(PM):,} rows)")

    print("[INFO] Aggregating weekly global…")
    weekly = aggregate_weekly(PM, by_surface=False)
    weekly_out = out_root / "tfc_weekly.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[INFO] Saved weekly TFC → {weekly_out} ({len(weekly):,} rows)")

    print("[INFO] Aggregating weekly by surface…")
    weekly_surf = aggregate_weekly(PM, by_surface=True)
    weekly_surf_out = out_root / "tfc_weekly_surface.csv"
    weekly_surf.to_csv(weekly_surf_out, index=False)
    print(f"[INFO] Saved weekly surface TFC → {weekly_surf_out} ({len(weekly_surf):,} rows)")

    # Quick coverage summary
    if not weekly.empty:
        print(f"[INFO] Coverage: {int(weekly['iso_year'].min())}-{int(weekly['iso_year'].max())} | Players: {weekly['player_id'].nunique()}")

if __name__ == "__main__":
    main()
