# index_cpi_weekly.py
"""
Weekly CPI (Top-50 Players, 1991–2025)
Includes both tournament-level and surface weighting.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


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
    df["has_tb"] = _detect_tiebreak(df["score"]).astype(int)
    df["deciding_set"] = _deciding_set(df["score"], df["best_of"]).astype(int)
    df["bp_faced"] = pd.to_numeric(df["bp_faced"], errors="coerce").fillna(0)
    df["high_bp_pressure"] = (df["bp_faced"] >= 4).astype(int)
    df["tier_w"] = df["tourney_level"].apply(_tier_weight)
    df["surf_w"] = df["surface"].apply(_surface_weight)
    df["combined_w"] = df["tier_w"] * df["surf_w"]
    df["PIF"] = (
        (0.45*df["has_tb"] + 0.35*df["deciding_set"] + 0.15*df["high_bp_pressure"])
        * df["combined_w"]
    ).clip(0, 1.5)
    df["is_pressure"] = ((df["has_tb"]==1)|(df["deciding_set"]==1)|(df["high_bp_pressure"]==1)).astype(int)
    return df


def compute_weekly_weighted_cpi(df):
    def calc(sub):
        wr_norm = sub.loc[sub["is_pressure"]==0, "label"].mean()
        wr_press = sub.loc[sub["is_pressure"]==1, "label"].mean()
        cpi = wr_press - wr_norm
        weighted = ((sub["label"]-sub["label"].mean()) * sub["PIF"]).mean()
        return pd.Series({
            "matches": len(sub),
            "press_matches": int(sub["is_pressure"].sum()),
            "wr_norm": wr_norm,
            "wr_press": wr_press,
            "CPI": cpi,
            "CPI_weighted": weighted,
        })
    return df.groupby(["player_id","player_name","gender","week"], as_index=False).apply(calc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    master["tourney_date"] = pd.to_datetime(master["tourney_date"], errors="coerce")

    # winner+loser flattening
    W = pd.DataFrame({
        "date": master["tourney_date"],
        "week": master["tourney_date"].dt.to_period("W-MON").dt.start_time,
        "player_id": master["winner_id"],
        "player_name": master["winner_name"],
        "gender": master["gender"],
        "tourney_level": master.get("tourney_level_norm", master.get("tourney_level","")),
        "surface": master["surface"],
        "best_of": master["best_of"],
        "score": master["score"],
        "bp_faced": master.get("w_bpFaced"),
        "label": 1,
    })
    L = W.copy()
    L["player_id"], L["player_name"], L["label"], L["bp_faced"] = (
        master["loser_id"], master["loser_name"], 0, master.get("l_bpFaced"),
    )
    df = pd.concat([W, L], ignore_index=True)

    df = build_pressure_flags(df)
    result = compute_weekly_weighted_cpi(df)

    out = Path(args.out_root) / "cpi_weekly_weighted_surface.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Saved weekly weighted CPI (tournament + surface) → {out}")


if __name__ == "__main__":
    main()


import pandas as pd
df = pd.read_csv(r"D:\Tennis\data\indexes\cpi_weighted_surface.csv")
print(df.columns)
print(df.head())
print(df["year"].unique()[:10])