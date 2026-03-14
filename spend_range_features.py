import numpy as np
import pandas as pd

def add_spend_range_features(
    snap: pd.DataFrame,
    S_EOM_min,
    S_EOM_max,
    gamma_map: pd.Series | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Adds scenario-conditioned features and then Min-Max normalizes numeric feature columns in-place
    (overrides values; no *_norm columns).

    Expects snap has at least:
      snapshot_day, days_in_month, days_remaining, R_MTD, S_MTD, spend_7d, spend_14d, rev_7d, rev_14d,
      spend_7d_std, rev_7d_std, game_id, month
    And either:
      - snap['gamma_eom'] exists, OR gamma_map provided (indexed by game_id).
    """
    x = snap.copy()

    # ---------- gamma ----------
    if "gamma_eom" not in x.columns:
        if gamma_map is None:
            raise ValueError("gamma_eom missing: provide snap['gamma_eom'] or gamma_map (indexed by game_id).")
        x["gamma_eom"] = x["game_id"].map(gamma_map).astype(float)

    if x["gamma_eom"].isna().any():
        missing = x.loc[x["gamma_eom"].isna(), "game_id"].unique()
        raise ValueError(f"Missing gamma_eom for game_ids: {missing}")

    # ---------- scenario inputs (scalar or vector) ----------
    x["S_EOM_min"] = S_EOM_min
    x["S_EOM_max"] = S_EOM_max

    # ---------- time mechanics ----------
    x["fraction_elapsed"] = x["snapshot_day"] / np.maximum(1.0, x["days_in_month"])

    # ---------- base ROAS / gaps ----------
    x["ROAS_MTD"] = x["R_MTD"] / np.maximum(eps, x["S_MTD"])
    x["ROAS_MTD_log"] = np.log1p(x["R_MTD"]) - np.log1p(x["S_MTD"])

    x["gamma_gap_mtd"] = x["ROAS_MTD"] - x["gamma_eom"]
    x["gamma_gap_mtd_log"] = (np.log((x["R_MTD"] + eps) / (x["S_MTD"] + eps))- np.log(x["gamma_eom"] + eps))

    x["rev_gap_to_gamma_mtd"] = x["R_MTD"] - x["gamma_eom"] * x["S_MTD"]
    x["ROAS_gap_norm"] = (x["ROAS_MTD"] - x["gamma_eom"]) / np.maximum(eps, x["gamma_eom"])
    x["expected_rev_needed"] = x["gamma_eom"] * x["S_MTD"]
    x["rev_gap_mtd"] = x["expected_rev_needed"] - x["R_MTD"]

    # ---------- overspend flags ----------
    x["overspent_min"] = (x["S_MTD"] > x["S_EOM_min"]).astype(int)
    x["overspent_max"] = (x["S_MTD"] > x["S_EOM_max"]).astype(int)

    # ---------- MIN scenario ----------
    x["S_rem_min_raw"] = x["S_EOM_min"] - x["S_MTD"]
    x["S_rem_min"] = np.maximum(eps, x["S_rem_min_raw"])
    x["pct_budget_spent_min"] = x["S_MTD"] / np.maximum(eps, x["S_EOM_min"])

    x["ROAS_req_rem_min"] = (x["gamma_eom"] * x["S_EOM_min"] - x["R_MTD"]) / x["S_rem_min"]

    x["S_target_to_date_min"] = x["S_EOM_min"] * x["fraction_elapsed"]
    x["S_dev_target_to_date_min"] = x["S_MTD"] - x["S_target_to_date_min"]

    # ---------- MAX scenario ----------
    x["S_rem_max_raw"] = x["S_EOM_max"] - x["S_MTD"]
    x["S_rem_max"] = np.maximum(eps, x["S_rem_max_raw"])
    x["pct_budget_spent_max"] = x["S_MTD"] / np.maximum(eps, x["S_EOM_max"])

    x["ROAS_req_rem_max"] = (x["gamma_eom"] * x["S_EOM_max"] - x["R_MTD"]) / x["S_rem_max"]

    x["S_target_to_date_max"] = x["S_EOM_max"] * x["fraction_elapsed"]
    x["S_dev_target_to_date_max"] = x["S_MTD"] - x["S_target_to_date_max"]

    # ---------- Acceleration ----------
    x["spend_accel_ratio"] = x["spend_7d"]/ (x["spend_14d"]+eps)
    x["rev_accel_ratio"] = x["rev_7d"]/ (x["rev_14d"] + eps)

    x["spend_accel_log"] = np.log1p(x["spend_7d"]) - np.log1p(x["spend_14d"])
    x["rev_accel_log"] = np.log1p(x["rev_7d"]) - np.log1p(x["rev_14d"])

    x["spend_accel_abs"] = x["spend_7d"] - x["spend_14d"]
    x["rev_accel_abs"] = x["rev_7d"] - x["rev_14d"]

    x["roas_velocity"] = (x["rev_7d"] / (eps + x["spend_7d"])) - (x["rev_14d"] / (eps + x["spend_14d"]))

    # ---------- Stability / Volatility ----------
    x["rev_cv_7d"] = np.log1p(x["rev_7d_std"]) - np.log1p(x["rev_7d"])
    x["spend_cv_7d"] = np.log1p(x["spend_7d_std"]) - np.log1p(x["spend_7d"])

    # ---------- Time level features ----------
    x["month_dt"] = pd.to_datetime(x["month"])
    x["month_num"] = x["month_dt"].dt.month

    x["month_sin"] = np.sin(2 * np.pi * x["month_num"] / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * x["month_num"] / 12.0)
    x["month_sin_interact"] = x["month_sin"] * x["gamma_gap_mtd"]

    x["late_gap"] = x["gamma_gap_mtd"] * x["fraction_elapsed"]
    x["late_gap_log"] = x["gamma_gap_mtd_log"] * x["fraction_elapsed"]

    x["gap_pacing"] = x["gamma_gap_mtd"] * x["spend_7d_std"]
    x["gap_pacing_log"] = x["gamma_gap_mtd_log"] * x["spend_7d_std"]

    x["gap_per_remaining_day"] = x["gamma_gap_mtd"] / np.maximum(1.0, x["days_remaining"])
    x["gap_per_remaining_day_log"] = x["gamma_gap_mtd_log"] / np.maximum(1.0, x["days_remaining"])

    x["spend_scale"] = np.log1p(x["S_MTD"])


    # -------------------------------------------------
    # Cohort curve acceleration features
    # -------------------------------------------------

    x["rpi_accel_d3_d1"] = x["rpi_d3_wmean"] - x["rpi_d1_wmean"]
    x["rpi_accel_d7_d3"] = x["rpi_d7_wmean"] - x["rpi_d3_wmean"]

    # Convexity: is growth accelerating or flattening?
    x["curve_convexity"] = (
        (x["rpi_d7_wmean"] - x["rpi_d3_wmean"])
        - (x["rpi_d3_wmean"] - x["rpi_d1_wmean"])
    )

    # -------------------------------------------------
    # Retention slope features
    # -------------------------------------------------

    x["ret_drop_d3_d1"] = x["ret_d3_wmean"] - x["ret_d1_wmean"]
    x["ret_drop_d7_d3"] = x["ret_d7_wmean"] - x["ret_d3_wmean"]

    x["rpi_ratio_d3_d1"] = x["rpi_d3_wmean"] / (x["rpi_d1_wmean"] + eps)
    x["rpi_ratio_d7_d3"] = x["rpi_d7_wmean"] / (x["rpi_d3_wmean"] + eps)

    # -------------------------------------------------
    # Curve vs contract interaction
    # -------------------------------------------------

    x["curve_vs_gamma"] = x["rpi_d7_wmean"] / (x["gamma_eom"] + eps)

    # -------------------------------------------------
    # Time-adjusted curve strength
    # -------------------------------------------------

    x["curve_strength_adj"] = x["rpi_d7_wmean"] * x["fraction_elapsed"]

    # -------------------------------------------------
    # Stability / confidence flags
    # -------------------------------------------------

    x["low_weight_flag"] = (x["cohort_spend_weight_sum"] < 1e-6).astype(int)

    # -------------------------------------------------
    # Missing indicators (important for tree models)
    # -------------------------------------------------

    cohort_cols = [
        "rpi_d1_wmean", "rpi_d3_wmean", "rpi_d7_wmean",
        "ret_d1_wmean", "ret_d3_wmean", "ret_d7_wmean",
        "cohort_age_wmean", "spend_share_last_3_cohorts"
    ]
    x["cohort_capital_interaction"] = x["rpi_d7_wmean"] * np.log1p(x["S_MTD"])
    x["cohort_roas_proxy_d7"] = x["rpi_d7_wmean"] / (x["gamma_eom"] + eps)
    x["recent_spend_curve_risk"] = x["spend_share_last_3_cohorts"] * x["rpi_d3_wmean"]
    for c in cohort_cols:
        x[f"{c}_isna"] = x[c].isna().astype(int)

    # -------------------------------------------------
    # Final numeric safety cleanup
    # -------------------------------------------------

    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    return x

