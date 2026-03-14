import numpy as np
import pandas as pd

def add_game_breach_rate_sd(
    df_all: pd.DataFrame,
    split_month: str,
    *,
    game_col: str = "game_id",
    sd_col: str = "snapshot_day",
    month_col: str = "month",
    target_col: str = "roas_breach",
    out_col: str = "game_breach_rate_sd",
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """
    Adds two columns:

        game_breach_rate_sd
        n_game_sd

    game_breach_rate_sd =
        P(breach | game_id, snapshot_day)
        estimated on pre-split data and smoothed toward the
        snapshot_day baseline.

    n_game_sd =
        number of historical cohorts used for that (game_id, snapshot_day).

    For unseen (game_id, snapshot_day):
        falls back to snapshot_day mean, then global mean.
    """

    df = df_all.copy()
    split_ts = pd.Timestamp(split_month)

    # -------- training reference (pre split only) --------
    ref = df[pd.to_datetime(df[month_col]) < split_ts].copy()
    ref[sd_col] = ref[sd_col].astype(int)

    if len(ref) == 0:
        raise ValueError("No pre-split data available to compute game priors.")

    y = ref[target_col].astype(int)

    # -------- snapshot_day baseline --------
    sd_stats = ref.groupby(sd_col)[target_col].agg(
        sd_mean="mean",
        sd_n="size"
    )

    sd_mean = sd_stats["sd_mean"]
    global_mean = float(y.mean())

    # -------- game + snapshot_day stats --------
    gsd_stats = ref.groupby([game_col, sd_col])[target_col].agg(
        gsd_mean="mean",
        n_game_sd="size"
    )

    # -------- join stats back to full dataframe --------
    df[sd_col] = df[sd_col].astype(int)

    df = df.join(gsd_stats, on=[game_col, sd_col])
    df = df.join(sd_mean.rename("sd_mean"), on=sd_col)

    # -------- fallback handling --------
    df["sd_mean"] = df["sd_mean"].fillna(global_mean)

    df["n_game_sd"] = df["n_game_sd"].fillna(0).astype(int)
    df["gsd_mean"] = df["gsd_mean"].fillna(df["sd_mean"])

    # -------- hierarchical smoothing --------
    n = df["n_game_sd"].astype(float)

    df[out_col] = (
        n * df["gsd_mean"] + smoothing * df["sd_mean"]
    ) / (n + smoothing)

    # -------- cleanup temporary columns --------
    df.drop(columns=["gsd_mean", "sd_mean"], inplace=True)

    return df
