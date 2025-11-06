"""
Build ML-ready meta datasets for multiple eras and export Torch-ready tensors.

Custom periods:
1. FULL_PERIOD      = All available years
2. ERA_1991_2025    = 1991–2025
3. ERA_2009_2025    = 2009–2025
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

try:
    import torch
except ImportError:
    torch = None


def _ensure_dirs(base_out: Path):
    (base_out / "meta").mkdir(parents=True, exist_ok=True)
    (base_out / "ml").mkdir(parents=True, exist_ok=True)

"""
def _build_long_form(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing tournament tier column
    if "tourney_level_norm" not in df.columns:
        if "tourney_level" in df.columns:
            df["tourney_level_norm"] = df["tourney_level"]
        else:
            df["tourney_level_norm"] = "Unknown"

    if "gender" not in df.columns:
        df["gender"] = "ATP"  # fallback

    # Winner side
    df_w = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], errors="coerce"),
        "tourney_id": df["tourney_id"],
        "tourney_name": df["tourney_name"],
        "surface": df["surface"],
        "tourney_level_norm": df["tourney_level_norm"],
        "gender": df["gender"],
        "best_of": df.get("best_of"),
        "minutes": df.get("minutes"),
        "round": df.get("round", "Unknown"),
        "score": df.get("score"),  # <-- ADD THIS LINE
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "player_rank": df["winner_rank"],
        "player_age": df["winner_age"],
        "player_ht": df["winner_ht"],
        "opp_id": df["loser_id"],
        "opp_name": df["loser_name"],
        "opp_rank": df["loser_rank"],
        "label_win": 1,
    })

    # Loser side
    df_l = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], errors="coerce"),
        "tourney_id": df["tourney_id"],
        "tourney_name": df["tourney_name"],
        "surface": df["surface"],
        "tourney_level_norm": df["tourney_level_norm"],
        "gender": df["gender"],
        "best_of": df.get("best_of"),
        "minutes": df.get("minutes"),
        "round": df.get("round", "Unknown"),
        "player_id": df["loser_id"],
        "player_name": df["loser_name"],
        "player_rank": df["loser_rank"],
        "player_age": df["loser_age"],
        "player_ht": df["loser_ht"],
        "opp_id": df["winner_id"],
        "opp_name": df["winner_name"],
        "opp_rank": df["winner_rank"],
        "w_se": df.get("l_se"),
        "l_se": df.get("w_se"),
        "label_win": 0,
    })

    long_df = pd.concat([df_w, df_l], ignore_index=True)
    long_df = long_df.dropna(subset=["date"])
    long_df = long_df.sort_values(["player_id", "date"], kind="mergesort").reset_index(drop=True)
    return long_df

"""
def _build_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a long-form dataframe with ALL master columns preserved."""

    df = df.copy()

    if "tourney_level_norm" not in df.columns:
        df["tourney_level_norm"] = df.get("tourney_level", "Unknown")

    if "gender" not in df.columns:
        df["gender"] = "ATP"

    # Determine winner/loser columns dynamically
    winner_prefix = [c for c in df.columns if c.startswith("winner_")]
    loser_prefix = [c for c in df.columns if c.startswith("loser_")]

    # Winner-side view (rename winner→player, loser→opp)
    df_w = df.rename(columns={
        **{c: c.replace("winner_", "player_") for c in winner_prefix},
        **{c: c.replace("loser_", "opp_") for c in loser_prefix}
    }).copy()
    df_w["label_win"] = 1

    # Loser-side view (rename loser→player, winner→opp)
    df_l = df.rename(columns={
        **{c: c.replace("loser_", "player_") for c in loser_prefix},
        **{c: c.replace("winner_", "opp_") for c in winner_prefix}
    }).copy()
    df_l["label_win"] = 0

    # Merge both into a single dataframe
    long_df = pd.concat([df_w, df_l], ignore_index=True)

    # Standardize the main date column
    if "tourney_date" in long_df.columns:
        long_df["date"] = pd.to_datetime(long_df["tourney_date"], errors="coerce")
    else:
        long_df["date"] = pd.NaT

    # Keep column order stable: date first, label last
    date_first = ["date"]
    label_last = ["label_win"]
    middle_cols = [c for c in long_df.columns if c not in date_first + label_last]
    long_df = long_df[date_first + middle_cols + label_last]

    return long_df


def _feature_engineer(long_df: pd.DataFrame):
    """Create ML features with basic imputations."""
    df = long_df.copy()

    df["rank_diff"] = (df["player_rank"].fillna(2000) - df["opp_rank"].fillna(2000)).astype(float)
    df["age"] = df["player_age"].fillna(df["player_age"].median())
    df["height_cm"] = df["player_ht"].fillna(df["player_ht"].median())
    df["minutes"] = df["minutes"].fillna(0)
    df["best_of"] = df["best_of"].fillna(3)
    df["se"] = df["w_se"].astype(float).fillna(df["w_se"].median())

    surf_dum = pd.get_dummies(df["surface"].fillna("Unknown"), prefix="surf")
    gen_dum = pd.get_dummies(df["gender"].fillna("UNK"), prefix="gen")
    tier_dum = pd.get_dummies(df["tourney_level_norm"].fillna("Unknown"), prefix="tier")

    df = pd.concat([df, surf_dum, gen_dum, tier_dum], axis=1)

    feature_cols = [
        "rank_diff", "age", "height_cm", "minutes", "best_of", "se"
    ] + list(surf_dum.columns) + list(gen_dum.columns) + list(tier_dum.columns)

    df["label"] = df["label_win"].astype(int)
    return df, feature_cols


def _cut_periods(df: pd.DataFrame):
    """Custom periods: all, 1991–2025, and 2009–2025."""
    df = df.copy()
    df["year"] = df["date"].dt.year

    cuts = {
        "FULL_PERIOD": df.copy(),
        "ERA_1991_2025": df[(df["year"] >= 1991) & (df["year"] <= 2025)].copy(),
        "ERA_2009_2025": df[(df["year"] >= 2009) & (df["year"] <= 2025)].copy(),
    }

    meta = {
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "rows_total": int(len(df)),
        "rows_by_period": {k: int(len(v)) for k, v in cuts.items()}
    }
    return cuts, meta


def _export_parquet_and_tensors(cuts, feature_cols, out_root):
    meta_dir = out_root / "meta"
    ml_dir = out_root / "ml"

    for period, d in cuts.items():
        if d.empty:
            warnings.warn(f"[{period}] empty, skipping.")
            continue

        keep_cols = ["date", "tourney_id", "tourney_name", "surface",
                     "tourney_level_norm", "gender", "player_id",
                     "player_name", "opp_id", "opp_name", "label"] + feature_cols
        d[keep_cols].to_parquet(meta_dir / f"meta_{period}.parquet", index=False)

        if torch is not None:
            X = d[feature_cols].astype(np.float32).values
            y = d["label"].astype(np.float32).values.reshape(-1, 1)
            payload = {
                "X": torch.from_numpy(X),
                "y": torch.from_numpy(y),
                "feature_names": feature_cols,
                "n_samples": len(d),
                "n_features": len(feature_cols),
                "period": period
            }
            torch.save(payload, ml_dir / f"tensors_{period}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="Path to master parquet")
    ap.add_argument("--out_root", required=True, help="Output root folder")
    args = ap.parse_args()

    master_path = Path(args.master)
    out_root = Path(args.out_root)
    _ensure_dirs(out_root)

    df = pd.read_parquet(master_path)
    # --- FIX: Convert numeric tourney_date like 19901231 → datetime ---
    if "tourney_date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["tourney_date"]):
            df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    # -----------------------------------------------------------------

    long_df = _build_long_form(df)
    feat_df, feature_cols = _feature_engineer(long_df)
    cuts, meta = _cut_periods(feat_df)
    _export_parquet_and_tensors(cuts, feature_cols, out_root)

    with open(out_root / "meta" / "meta_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Meta datasets built successfully:")
    for k, v in meta["rows_by_period"].items():
        print(f"  {k}: {v:,} rows")
    print(f"\nSaved under {out_root / 'meta'} and {out_root / 'ml'}")


if __name__ == "__main__":
    main()


