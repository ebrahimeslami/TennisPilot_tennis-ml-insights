# index_cpi.py
"""
Clutch Pressure Index (CPI)
===========================

Purpose
-------
This module computes the **Clutch Pressure Index (CPI)** — a quantitative measure
of how well tennis players perform under high-pressure situations.

CPI integrates both statistical and machine-learning approaches to evaluate
“mental toughness” and performance stability across surfaces, tournament levels,
and time periods.

Concept
-------
Players are often judged not only by their average win rate but by how
they perform in decisive moments. CPI compares a player's win rate in
high-pressure points or matches to their baseline performance.

Mathematically:

    CPI = (Win Rate under Pressure) − (Win Rate in Normal Conditions)

and a *weighted* version adjusts for tournament level and surface:

    CPI_weighted = Σ(pressure_factor × tournament_weight × surface_weight × result)

Values interpretation:
    • Positive CPI  → performs **better than expected** in pressure moments
    • Zero CPI      → performs **as expected**
    • Negative CPI  → performs **worse under pressure**

Pressure is defined by:
    • Matches with ≥1 tiebreak set
    • Matches decided in final set (Bo3 or Bo5)
    • Matches where player faced ≥4 break points
    • Deep rounds (QF–Final) and higher tournament tiers receive higher weights

Outputs
-------
The script produces the following files (examples):

    D:\Tennis\data\indexes\
        cpi_weighted_surface.csv             # Annual CPI per player
        cpi_weekly_weighted_surface.parquet  # Weekly CPI time-series (Top-50)
        models\rf_cpi_baseline.joblib        # Trained RandomForest baseline (optional)

Columns:
    player_id        : unique identifier
    player_name      : player’s full name
    year             : calendar year of aggregation
    matches          : total matches used
    press_matches    : subset of high-pressure matches
    wr_norm          : win rate in non-pressure matches
    wr_press         : win rate in pressure matches
    CPI              : raw difference (wr_press − wr_norm)
    CPI_weighted     : CPI adjusted by tournament and surface weights

Usage
-----
Command-line example:

    python index_cpi.py --master "D:\Tennis\data\master\tennis_master_1991.parquet" \
                        --out_root "D:\Tennis\data\indexes"

Optional flag for ML baseline:

    python index_cpi.py ... --train_rf

This builds a RandomForest model on normal-pressure data and predicts
expected win probability under pressure; residuals become CPI estimates.

Interpretation Example
----------------------
Suppose Novak Djokovic’s CPI_weighted = +0.18 for 2021.
It means his win rate under pressure situations (tiebreaks, deciding sets, etc.)
was 18 % higher than expected given his baseline and match conditions.

Similarly, a player with CPI_weighted = −0.10 tends to underperform
in clutch moments despite similar baseline strength.

Machine-Learning Extension
--------------------------
The CPI can serve as:
    • A feature in broader performance models (momentum, fatigue, prediction)
    • A target variable for coaching analytics (mental toughness tracking)
    • A benchmark for fan engagement (pressure performance ranking)

Author
------
Ebrahim Eslami (EnviroPilot.ai)
2025
"""

"""
Clutch Pressure Index (CPI)
---------------------------
Now includes weighted scores based on both tournament level and surface type.

Tournament Weights:
    Grand Slam ............ 1.00
    Finals/Masters ........ 0.85–0.80
    ATP500 ................ 0.60
    ATP250 ................ 0.45
    Challenger/Future ..... 0.30
    Default ............... 0.40

Surface Weights:
    Grass ................. 1.00  (high variability, fewer events)
    Clay .................. 0.85  (physically demanding, long rallies)
    Hard .................. 0.70  (more consistent surface)
    Carpet ................ 0.60
    Default ............... 0.75
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _tier_weight(tier):
    t = str(tier).lower()
    if "grand" in t or t.strip() in ["g", "slam"]:
        return 1.00
    if "final" in t:
        return 0.85
    if "master" in t or t.strip() == "m":
        return 0.80
    if "500" in t:
        return 0.60
    if "250" in t:
        return 0.45
    if "challenger" in t or t.strip() == "c":
        return 0.30
    return 0.40


def _surface_weight(surface):
    s = str(surface).lower()
    if "grass" in s:
        return 1.00
    if "clay" in s:
        return 0.85
    if "hard" in s:
        return 0.70
    if "carpet" in s:
        return 0.60
    return 0.75


def _detect_tiebreak(score):
    s = score.fillna("").astype(str)
    return s.str.contains("7-6") | s.str.contains("6-7")


def _count_sets(score):
    if not isinstance(score, str) or not score:
        return 0
    return len([seg for seg in score.split() if any(ch.isdigit() for ch in seg)])


def _deciding_set(score, best_of):
    nsets = score.fillna("").astype(str).apply(_count_sets)
    bo = best_of.fillna(3).astype(int)
    return ((bo == 3) & (nsets >= 3)) | ((bo == 5) & (nsets >= 5))


def build_pressure_flags(df):
    df = df.copy()
    df["has_tb"] = _detect_tiebreak(df["score"]).astype(int)
    df["deciding_set"] = _deciding_set(df["score"], df.get("best_of", 3)).astype(int)
    df["bp_faced"] = pd.to_numeric(df["bp_faced"], errors="coerce").fillna(0)
    df["high_bp_pressure"] = (df["bp_faced"] >= 4).astype(int)
    df["tier_w"] = df["tourney_level"].apply(_tier_weight)
    df["surf_w"] = df["surface"].apply(_surface_weight)
    # combine weights multiplicatively (so both affect intensity)
    df["combined_w"] = df["tier_w"] * df["surf_w"]
    df["PIF"] = (
        (0.45 * df["has_tb"] + 0.35 * df["deciding_set"] + 0.15 * df["high_bp_pressure"])
        * df["combined_w"]
    ).clip(0, 1.5)
    df["is_pressure"] = (
        (df["has_tb"] == 1) | (df["deciding_set"] == 1) | (df["high_bp_pressure"] == 1)
    ).astype(int)
    return df


def compute_weighted_cpi(df):
    """Weighted CPI per player-year."""
    def calc(sub):
        wr_norm = sub.loc[sub["is_pressure"]==0, "label"].mean()
        wr_press = sub.loc[sub["is_pressure"]==1, "label"].mean()
        cpi = wr_press - wr_norm
        weighted = ((sub["label"] - sub["label"].mean()) * sub["PIF"]).mean()
        return pd.Series({
            "matches": len(sub),
            "press_matches": int(sub["is_pressure"].sum()),
            "wr_norm": wr_norm,
            "wr_press": wr_press,
            "CPI": cpi,
            "CPI_weighted": weighted,
        })
    return df.groupby(["player_id","player_name","year"], as_index=False).apply(calc)


def _long_from_master(df):
    """Convert match-level data to player-level (winner + loser)."""
    import pandas as pd

    # --- Guaranteed fix for int YYYYMMDD format ---
    # Convert all values to string before parsing (avoids int32 issues)
    df["tourney_date"] = df["tourney_date"].astype("int64").astype(str)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # Drop invalids and filter for 1991+
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 1991]

    df["year"] = df["tourney_date"].dt.year

    print(f"[INFO] Date range: {df['tourney_date'].min()} → {df['tourney_date'].max()}")
    print(f"[INFO] Years detected: {sorted(df['year'].unique())[:10]} ...")

    # ---- continue with winner/loser transformation ----
    W = pd.DataFrame({
        "date": df["tourney_date"],
        "year": df["year"],
        "tourney_id": df["tourney_id"],
        "tourney_level": df.get("tourney_level_norm", df.get("tourney_level", "")),
        "surface": df["surface"],
        "round": df["round"],
        "gender": df.get("gender", "M"),
        "best_of": df["best_of"],
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "player_rank": df["winner_rank"],
        "opp_id": df["loser_id"],
        "opp_rank": df["loser_rank"],
        "score": df["score"],
        "bp_faced": df.get("w_bpFaced"),
        "label": 1,
    })

    L = W.copy()
    L["player_id"], L["player_name"], L["player_rank"], L["opp_id"], L["opp_rank"], L["label"] = (
        df["loser_id"], df["loser_name"], df["loser_rank"], df["winner_id"], df["winner_rank"], 0,
    )

    df_long = pd.concat([W, L], ignore_index=True).dropna(subset=["date", "player_id", "opp_id"])

    print(f"[INFO] Date range (long): {df_long['date'].min()} → {df_long['date'].max()}")
    print(f"[INFO] Years detected (long): {sorted(df_long['year'].unique())[:10]} ...")

    return df_long



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    master["tourney_date"] = pd.to_datetime(master["tourney_date"], errors="coerce")

    df_long = _long_from_master(master)
    df_long = build_pressure_flags(df_long)

    result = compute_weighted_cpi(df_long)
    out = Path(args.out_root) / "cpi_weighted_surface.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved tournament + surface weighted CPI → {out}")


if __name__ == "__main__":
    main()

