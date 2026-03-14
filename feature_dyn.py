import numpy as np
import pandas as pd

def compute_dyn_features_from_daily(
    df_daily: pd.DataFrame,
    snapshot_days=(7, 14, 21, 28),
    eps: float = 1e-9,
) -> pd.DataFrame:
    keys = ["source_system", "client_id", "game_id", "cohort", "month", "snapshot_day"]
    need = ["source_system", "client_id", "game_id", "cohort", "month", "d", "spend_d", "rev_d"]
    missing = [c for c in need if c not in df_daily.columns]
    if missing:
        raise KeyError(f"df_daily missing columns: {missing}")

    dfd = df_daily.copy()

    # keep your existing datetime structure
    dfd["month"] = pd.to_datetime(dfd["month"])
    dfd["d"] = pd.to_datetime(dfd["d"])

    # robust day-of-cohort within the month (1..31), aligned to `month`
    month_start = dfd["month"].dt.to_period("M").dt.to_timestamp()
    d_norm = dfd["d"].dt.to_period("D").dt.to_timestamp()
    dfd["day"] = (d_norm - month_start).dt.days + 1

    # drop rows that don't land in the same month bucket cleanly
    dfd = dfd[dfd["day"].between(1, 31)]
    dfd["day"] = dfd["day"].astype(int)

    grp_cols = ["source_system", "client_id", "game_id", "cohort", "month"]
    out = []

    for sd in snapshot_days:
        # use all daily rows up to snapshot day
        dd = dfd[dfd["day"] <= sd].copy()
        if dd.empty:
            continue

        last7_lo, last7_hi = sd - 6, sd
        prev7_lo, prev7_hi = sd - 13, sd - 7
        last3_lo, last3_hi = sd - 2, sd
        prev3_lo, prev3_hi = sd - 5, sd - 3

        def rng(g: pd.DataFrame, lo: int, hi: int, col: str) -> pd.Series:
            m = (g["day"] >= lo) & (g["day"] <= hi)
            return g.loc[m, col]

        rows = []
        for k, g in dd.groupby(grp_cols, sort=False):
            spend_last7 = rng(g, last7_lo, last7_hi, "spend_d")
            spend_prev7 = rng(g, prev7_lo, prev7_hi, "spend_d")
            rev_last7   = rng(g, last7_lo, last7_hi, "rev_d")
            rev_prev7   = rng(g, prev7_lo, prev7_hi, "rev_d")

            spend_7d_sum = float(spend_last7.sum()) if len(spend_last7) else 0.0
            spend_prev7_sum = float(spend_prev7.sum()) if len(spend_prev7) else 0.0
            rev_7d_sum = float(rev_last7.sum()) if len(rev_last7) else 0.0
            rev_prev7_sum = float(rev_prev7.sum()) if len(rev_prev7) else 0.0

            spend_7d_std = float(spend_last7.std(ddof=0)) if len(spend_last7) else np.nan

            spend_mean = float(spend_last7.mean()) if len(spend_last7) else np.nan
            spend_max  = float(spend_last7.max()) if len(spend_last7) else np.nan
            spend_impulse = np.log1p(spend_max) - np.log1p(spend_mean) if np.isfinite(spend_mean) else np.nan

            spend_last3 = rng(g, last3_lo, last3_hi, "spend_d")
            spend_prev3 = rng(g, prev3_lo, prev3_hi, "spend_d")
            a = float(spend_last3.mean()) if len(spend_last3) else np.nan
            b = float(spend_prev3.mean()) if len(spend_prev3) else np.nan
            spend_slope_last3_vs_prev3 = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
            spend_decelerating = float(spend_slope_last3_vs_prev3 < 0) if np.isfinite(spend_slope_last3_vs_prev3) else np.nan

            spend_pacing_ratio_7d_to_prev7d = np.log1p(spend_7d_sum)- np.log1p(spend_prev7_sum)

            row = dict(zip(grp_cols, k if isinstance(k, tuple) else (k,)))
            row["snapshot_day"] = int(sd)
            row.update({
                "spend_7d_sum": spend_7d_sum,
                "spend_prev7_sum": spend_prev7_sum,
                "rev_7d_sum": rev_7d_sum,
                "rev_prev7_sum": rev_prev7_sum,
                "spend_7d_std_dyn": spend_7d_std,
                "spend_impulse": spend_impulse,
                "spend_slope_last3_vs_prev3": spend_slope_last3_vs_prev3,
                "spend_decelerating": spend_decelerating,
                "spend_pacing_ratio_7d_to_prev7d": spend_pacing_ratio_7d_to_prev7d,
            })
            rows.append(row)

        if rows:
            out.append(pd.DataFrame(rows))

    df_out = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keys)

    # --------------------------
    # Add Min-Max normalized features (per snapshot_day)
    # - creates <feature>_norm columns
    # - keeps original columns
    # - normalizes within each snapshot_day across all groups
    # - constants -> 0.0, all-NaN -> NaN
    # --------------------------
    if not df_out.empty:
        non_feature_cols = set(keys)
        drop = ["spend_impulse",  "spend_pacing_ratio_7d_to_prev7d","spend_decelerating",]
        feat_cols = [c for c in df_out.columns if (c not in non_feature_cols)
                     and (c not in drop)]

        def _minmax_per_group(s: pd.Series) -> pd.Series:
            s_num = pd.to_numeric(s, errors="coerce")
            mn = s_num.min(skipna=True)
            mx = s_num.max(skipna=True)
            if not np.isfinite(mn) or not np.isfinite(mx):
                return pd.Series(np.nan, index=s.index)
            rng = mx - mn
            if rng <= eps:
                # constant (or near-constant) -> 0.0 for finite values, NaN stays NaN
                out_s = pd.Series(np.nan, index=s.index)
                mask = np.isfinite(s_num.to_numpy())
                out_s.loc[mask] = 0.0
                return out_s
            return (s_num - mn) / (rng + eps)

        # compute normalized columns per snapshot_day
        for c in feat_cols:
            df_out[f"{c}_norm"] = df_out.groupby("snapshot_day", sort=False)[c].transform(_minmax_per_group)

    return df_out

