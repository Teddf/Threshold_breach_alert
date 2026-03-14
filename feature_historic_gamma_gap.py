import numpy as np
import pandas as pd

def add_gamma_gap_relative_to_game_history(
    df_all: pd.DataFrame,
    split_month: str,
    *,
    game_col: str = "game_id",
    sd_col: str = "snapshot_day",
    month_col: str = "month",
    gap_col: str = "gamma_gap_mtd",
    # choose baseline statistic
    baseline_stat: str = "median",   # "median" (recommended) or "mean"
    # smoothing to avoid noisy baselines with few cohorts
    smoothing_n: float = 20.0,
    out_cols_prefix: str = "gap_rel",
):
    """
    Adds three features:
      - {prefix}_centered: gap - baseline_gap(game, sd)   (safe default)
      - {prefix}_ratio:    gap / (baseline_gap + eps)    (optional; can be unstable near 0)
      - {prefix}_z:        (gap - baseline) / (mad + eps) robust z-score (recommended if you want scaling)

    Baselines are computed on PRE-SPLIT only (month < split_month) to avoid leakage.
    Unseen (game,sd): fall back to snapshot_day baseline; if unseen sd: global baseline.
    """
    df = df_all.copy()
    split_ts = pd.Timestamp(split_month)

    # pre-split reference (no leakage)
    ref = df[pd.to_datetime(df[month_col]) < split_ts].copy()
    ref[sd_col] = ref[sd_col].astype(int)

    # snapshot-day fallback baseline
    if baseline_stat == "median":
        sd_base = ref.groupby(sd_col)[gap_col].median().rename("sd_base")
        gsd_base = ref.groupby([game_col, sd_col])[gap_col].median().rename("gsd_base")
    elif baseline_stat == "mean":
        sd_base = ref.groupby(sd_col)[gap_col].mean().rename("sd_base")
        gsd_base = ref.groupby([game_col, sd_col])[gap_col].mean().rename("gsd_base")
    else:
        raise ValueError("baseline_stat must be 'median' or 'mean'")

    # counts for smoothing
    gsd_n = ref.groupby([game_col, sd_col])[gap_col].size().rename("gsd_n")

    # robust spread (MAD) for z-score (computed on train only)
    # MAD = median(|x - median(x)|)
    gsd_med = gsd_base
    gsd_mad = (
        ref.join(gsd_med, on=[game_col, sd_col])
           .assign(abs_dev=lambda d: (d[gap_col] - d["gsd_base"]).abs())
           .groupby([game_col, sd_col])["abs_dev"].median()
           .rename("gsd_mad")
    )
    sd_mad = (
        ref.join(sd_base, on=sd_col)
           .assign(abs_dev=lambda d: (d[gap_col] - d["sd_base"]).abs())
           .groupby(sd_col)["abs_dev"].median()
           .rename("sd_mad")
    )

    # merge baselines into full df
    df[sd_col] = df[sd_col].astype(int)
    df = df.join(gsd_base, on=[game_col, sd_col])
    df = df.join(gsd_n, on=[game_col, sd_col])
    df = df.join(sd_base, on=sd_col)
    df = df.join(gsd_mad, on=[game_col, sd_col])
    df = df.join(sd_mad, on=sd_col)

    # fallback chain: (game,sd) -> sd -> global
    global_base = float(ref[gap_col].median() if baseline_stat == "median" else ref[gap_col].mean())
    global_mad = float((ref[gap_col] - (ref[gap_col].median() if baseline_stat=="median" else ref[gap_col].mean())).abs().median())

    df["base_gap"] = df["gsd_base"]
    df.loc[df["base_gap"].isna(), "base_gap"] = df.loc[df["base_gap"].isna(), "sd_base"]
    df["base_gap"] = df["base_gap"].fillna(global_base)

    df["base_mad"] = df["gsd_mad"]
    df.loc[df["base_mad"].isna(), "base_mad"] = df.loc[df["base_mad"].isna(), "sd_mad"]
    df["base_mad"] = df["base_mad"].fillna(global_mad)

    # smoothing: baseline = (n*game_sd_base + k*sd_base)/(n+k)
    # if gsd missing, n is NaN -> treat as 0
    n = df["gsd_n"].fillna(0.0).astype(float)
    sd_base_full = df["sd_base"].fillna(global_base).astype(float)
    gsd_base_full = df["gsd_base"].fillna(sd_base_full).astype(float)
    smoothed_base = (n * gsd_base_full + smoothing_n * sd_base_full) / (n + smoothing_n)

    eps = 1e-6
    gap = df[gap_col].astype(float)

    df[f"{out_cols_prefix}_centered"] = gap - smoothed_base
    df[f"{out_cols_prefix}_ratio"] = gap / (smoothed_base.abs() + eps)  # uses abs to reduce sign issues
    df[f"{out_cols_prefix}_z"] = (gap - smoothed_base) / (df["base_mad"].abs() + eps)

    # cleanup helper cols (optional)
    df = df.drop(columns=[c for c in ["gsd_base","gsd_n","sd_base","gsd_mad","sd_mad","base_gap","base_mad"] if c in df.columns])

    return df

