import numpy as np
import pandas as pd
import shap

# -----------------------------
# Client-facing configuration
# -----------------------------

CLIENT_DRIVER_WHITELIST = [
    # gap / pacing
    "gamma_gap_mtd", "expected_rev_needed", "gap_per_remaining_day", "rev_gap_mtd",
    # revenue curve
    "rpi_d1_wmean", "rpi_d3_wmean", "rpi_d7_wmean",
    "rpi_ratio_d3_d1", "rpi_ratio_d7_d3",
    "rpi_accel_d3_d1", "rpi_accel_d7_d3", "curve_convexity",
    # retention
    "ret_d1_wmean", "ret_d3_wmean", "ret_d7_wmean", "ret_drop_d3_d1",
    # spend stress
    "spend_accel_log", "spend_pacing_ratio_7d_to_prev7d", "spend_slope_last3_vs_prev3", "spend_7d_std_dyn",
    # optional summary proxy
    "cohort_roas_proxy_d7",
]

# Friendly labels + units + formatting.
# tweak these labels without touching the logic.
FEATURE_META = {
    "gamma_gap_mtd": dict(label="Gap to EOM cumulative ROAS target (MTD)", unit="ROAS", fmt="{:+.3f}"),
    "expected_rev_needed": dict(label="minimum total Revenue needed to hit target", unit="$", fmt="{:,.0f}"),
    "rev_gap_mtd": dict(label="minimum remaining Revenue needed to hit target", unit="$", fmt="{:,.0f}"),
    "gap_per_remaining_day": dict(label="Required cumulative ROAS gap per day", unit="ROAS", fmt="{:,.0f}"),
    "late_gap": dict(label="Behind-schedule indicator (gap timing)", unit="ROAS", fmt="{:+.3f}"),

    "rpi_d1_wmean": dict(label="RPI at D1 (weighted)", unit="$/install", fmt="{:.3f}"),
    "rpi_d3_wmean": dict(label="RPI at D3 (weighted)", unit="$/install", fmt="{:.3f}"),
    "rpi_d7_wmean": dict(label="RPI at D7 (weighted)", unit="$/install", fmt="{:.3f}"),
    "rpi_ratio_d3_d1": dict(label="RPI maturation D3 vs D1", unit="ratio", fmt="{:.3f}"),
    "rpi_ratio_d7_d3": dict(label="RPI maturation D7 vs D3", unit="ratio", fmt="{:.3f}"),
    "rpi_accel_d3_d1": dict(label="RPI acceleration (D3-D1)", unit="$/install/day", fmt="{:+.4f}"),
    "rpi_accel_d7_d3": dict(label="RPI acceleration (D7-D3)", unit="$/install/day", fmt="{:+.4f}"),
    "curve_convexity": dict(label="Revenue curve shape (convexity)", unit="index", fmt="{:+.3f}"),

    "ret_d1_wmean": dict(label="Retention at D1 (weighted)", unit="%", fmt="{:.1%}"),
    "ret_d3_wmean": dict(label="Retention at D3 (weighted)", unit="%", fmt="{:.1%}"),
    "ret_d7_wmean": dict(label="Retention at D7 (weighted)", unit="%", fmt="{:.1%}"),
    "ret_drop_d3_d1": dict(label="Early retention drop (D3 vs D1)", unit="pp", fmt="{:+.2%}"),

    "spend_accel_log": dict(label="Spend acceleration (log)", unit="log", fmt="{:+.3f}"),
    "spend_pacing_ratio_7d_to_prev7d": dict(label="Spend pacing ratio (last 7d vs prior 7d)", unit="ratio", fmt="{:.2f}"),
    "spend_slope_last3_vs_prev3": dict(label="Spend slope change (last 3 vs prior 3)", unit="index", fmt="{:+.3f}"),
    "spend_7d_std_dyn": dict(label="Spend volatility (7d std)", unit="$", fmt="{:,.0f}"),

    "cohort_roas_proxy_d7": dict(label="Early ROAS proxy at D7", unit="ROAS", fmt="{:.3f}"),
}

# -----------------------------
# Helper functions
# -----------------------------

def _get_label_col(d: pd.DataFrame) -> str | None:
    for c in ["y_true", "roas_breach"]:
        if c in d.columns:
            return c
    return None

def _safe_float(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None

def _percentile_of_value(value: float, ref: np.ndarray) -> float | None:
    """
    Percentile rank of 'value' within ref (0..100). Returns None if ref too small.
    """
    if value is None:
        return None
    ref = np.asarray(ref)
    ref = ref[np.isfinite(ref)]
    if ref.size < 20:   # guard against noisy percentiles
        return None
    return float(100.0 * (ref <= value).mean())

def _format_value(feature: str, value):
    meta = FEATURE_META.get(feature)
    if meta is None:
        # fallback
        try:
            return str(value)
        except Exception:
            return ""
    fmt = meta.get("fmt", "{}")
    try:
        v = float(value)
        if np.isnan(v):
            return ""
        return fmt.format(v)
    except Exception:
        return str(value)

def _feature_label(feature: str) -> str:
    return FEATURE_META.get(feature, {}).get("label", feature)

def _feature_unit(feature: str) -> str | None:
    return FEATURE_META.get(feature, {}).get("unit")

def _build_reference_pool(
    *,
    df_all: pd.DataFrame,
    split_month: str,
    snapshot_day: int,
    game_id: str,
    feature_cols: list[str],
    month_col: str = "month",
) -> pd.DataFrame:
    m = (
        (df_all["snapshot_day"].astype(int) == int(snapshot_day)) &
        (df_all["game_id"] == game_id)
    )
    ref = df_all.loc[m].copy()

    if month_col in ref.columns:
        ref_month = pd.to_datetime(ref[month_col])
        ref = ref.loc[ref_month < pd.Timestamp(split_month)].copy()

    keep = ["roas_breach", "game_breach_rate_sd", "n_game_sd"] + [c for c in feature_cols if c in ref.columns]
    keep = [c for c in keep if c in ref.columns]
    return ref[keep].copy()

def _risk_bucket_from_margin(p_breach: float, threshold: float) -> dict:
    """
    Bucket purely from margin to your operational threshold.
    You can later replace with data-driven buckets without changing the API.
    """
    margin = p_breach - threshold
    if margin >= 0.10:
        bucket = "High"
    elif margin <= -0.10:
        bucket = "Low"
    else:
        bucket = "Medium"
    return {"bucket": bucket, "margin": float(margin)}

def _compute_client_metrics(row: pd.Series) -> dict:
    """
    Uses fraction_elapsed (as requested) for time context.
    """
    S = _safe_float(row.get("S_MTD"))
    R = _safe_float(row.get("R_MTD"))
    roas_mtd = None
    if S is not None and S > 0 and R is not None:
        roas_mtd = float(R / S)

    metrics = {
        "spend_mtd": S,
        "revenue_mtd": R,
        "roas_mtd": roas_mtd,
        "target_roas_eom": _safe_float(row.get("gamma_eom")),
        "roas_eom_actual_if_known": _safe_float(row.get("ROAS_EOM")),
        "fraction_elapsed": _safe_float(row.get("fraction_elapsed")),
        "revenue_needed_to_hit_target": _safe_float(row.get("expected_rev_needed")),
        # "required_revenue_per_remaining_day": _safe_float(row.get("exp")),
        "behind_schedule_signal": _safe_float(row.get("late_gap")),
        # curve + retention headline values
        "rpi_d1": _safe_float(row.get("rpi_d1_wmean")),
        "rpi_d3": _safe_float(row.get("rpi_d3_wmean")),
        "rpi_d7": _safe_float(row.get("rpi_d7_wmean")),
        "ret_d1": _safe_float(row.get("ret_d1_wmean")),
        "ret_d3": _safe_float(row.get("ret_d3_wmean")),
        "ret_d7": _safe_float(row.get("ret_d7_wmean")),
    }
    return metrics

def _recommend_actions(p_breach: float, threshold: float, metrics: dict, drivers_df: pd.DataFrame) -> list[str]:
    """
    Deterministic, conservative rules. Keep short; client-facing.
    """
    actions = []

    margin = p_breach - threshold
    late_gap = metrics.get("behind_schedule_signal")
    req_per_day = metrics.get("revenue_needed_to_hit_target")

    # helper: does a driver appear in top impacts
    top_feats = set(drivers_df["feature"].tolist()) if drivers_df is not None and len(drivers_df) else set()

    if margin >= 0.10:
        if "late_gap" in top_feats or (late_gap is not None and late_gap > 0):
            actions.append("Reduce spend pace until next snapshot unless performance improves (behind schedule vs target).")
        if "ret_d3_wmean" in top_feats or "ret_drop_d3_d1" in top_feats:
            actions.append("Investigate traffic quality and early funnel drivers (retention signals are pushing risk up).")
        if "rpi_accel_d7_d3" in top_feats or "curve_convexity" in top_feats:
            actions.append("Check whether monetization is stalling vs prior cohorts; adjust bids/geo mix if applicable.")
        if req_per_day is not None and req_per_day > 0:
            actions.append("Treat recovery as requiring unusually strong late-month monetization; plan scenarios accordingly.")
    elif margin <= -0.10:
        actions.append("Maintain pacing; continue monitoring on the next snapshot.")
        if "spend_pacing_ratio_7d_to_prev7d" in top_feats:
            actions.append("Avoid sudden spend ramp-ups that could change cohort mix; scale gradually.")
    else:
        actions.append("Hold pace and re-evaluate at the next snapshot; risk is near decision boundary.")
        if "late_gap" in top_feats:
            actions.append("Watch pacing vs target closely; small changes can move this cohort into high risk.")
    # keep concise
    return actions[:4]


def _compute_adjusted_probability(
    p_model: float,
    game_breach_rate_sd: float | None,
    n_game_sd: float | None = None,
    shrinkage: float = 20.0,
) -> tuple[float, float]:
    """
    Returns:
      p_adjusted, blend_weight
    """
    if game_breach_rate_sd is None:
        return float(p_model), 0.0

    n = 0.0 if n_game_sd is None else float(max(n_game_sd, 0.0))
    w = n / (n + shrinkage) if (n + shrinkage) > 0 else 0.0
    p_adj = (1.0 - w) * float(p_model) + w * float(game_breach_rate_sd)
    return float(p_adj), float(w)


def _risk_bucket_from_quantiles(
    p_value: float,
    ref_probs: np.ndarray,
    low_q: float = 0.20,
    high_q: float = 0.75,
) -> dict:
    """
    Buckets risk using the historical probability distribution.
    """
    ref = np.asarray(ref_probs, dtype=float)
    ref = ref[np.isfinite(ref)]
    '''
    if ref.size < 20:
        return {
            "bucket": None,
            "percentile": None,
            "q_low": None,
            "q_high": None,
        }
    '''
    q_low_val = float(np.quantile(ref, low_q))
    q_high_val = float(np.quantile(ref, high_q))
    percentile = float(100.0 * (ref <= p_value).mean())

    if p_value < q_low_val:
        bucket = "Low"
    elif p_value >= q_high_val:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "bucket": bucket,
        "percentile": percentile,
        "q_low": q_low_val,
        "q_high": q_high_val,
    }



def _risk_percentile(p_value: float, ref_probs: np.ndarray) -> float | None:
    ref = np.asarray(ref_probs, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 20:
        return None
    return float(100.0 * (ref <= p_value).mean())


# -----------------------------
# Main function (drop-in)
# -----------------------------

def explain_breach_for_client(
    *,
    df_all: pd.DataFrame,
    models_by_sd: dict,
    thresholds_by_sd: dict,
    feature_cols: list[str],
    split_month: str,                  # needed for clean historical baselines/percentiles (pre-split)
    game_id: str,
    snapshot_day: int,
    cohort_month: str | pd.Timestamp,  # cohort identifier in your df (column "cohort")
    k: int = 8,
    month: str | pd.Timestamp | None = None,  # only if you need to disambiguate
    driver_whitelist: list[str] | None = None,

    COHORT_COL: str = "cohort",
    GAME_COL: str = "game_id",
    SD_COL: str = "snapshot_day",
    MONTH_COL: str = "month",
    label_cols: tuple[str, ...] = ("y_true", "roas_breach"),
) -> dict:
    """
    Client-facing payload:
      - headline: p, threshold, margin, bucket (from margin)
      - historic_context: per-game baseline breach rate (pre-split) at same snapshot_day + sample size
      - metrics: key spend/rev/roas + target + gap metrics + curve/retention headline
      - drivers: top-k SHAP impacts (client whitelist), with formatted values + percentiles
      - actions: short rule-based suggestions
    """

    snapshot_day = int(snapshot_day)
    if snapshot_day not in models_by_sd:
        raise ValueError(f"models_by_sd missing model for snapshot_day={snapshot_day}")
    if snapshot_day not in thresholds_by_sd:
        raise ValueError(f"thresholds_by_sd missing threshold for snapshot_day={snapshot_day}")

    model = models_by_sd[snapshot_day]
    thr = float(thresholds_by_sd[snapshot_day])

    cohort_ts = pd.Timestamp(cohort_month)
    month_ts = pd.Timestamp(month) if month is not None else None

    # ---- filter single row ----
    mask = (
        (df_all[GAME_COL] == game_id) &
        (df_all[SD_COL].astype(int) == snapshot_day) &
        (pd.to_datetime(df_all[COHORT_COL]) == cohort_ts)
    )
    if month_ts is not None and MONTH_COL in df_all.columns:
        mask = mask & (pd.to_datetime(df_all[MONTH_COL]) == month_ts)

    df_row = df_all.loc[mask].copy()
    if df_row.empty:
        raise ValueError(
            f"No row found for game_id={game_id}, snapshot_day={snapshot_day}, {COHORT_COL}={cohort_ts.date()}"
            + (f", {MONTH_COL}={month_ts.date()}" if month_ts is not None else "")
        )
    if len(df_row) > 1:
        cols_show = [c for c in [GAME_COL, SD_COL, COHORT_COL, MONTH_COL] if c in df_row.columns]
        raise ValueError(
            f"Expected 1 row but found {len(df_row)}. Disambiguate by adding `month=`. "
            f"Sample keys:\n{df_row[cols_show].head(10).to_string(index=False)}"
        )
    row = df_row.iloc[0]
    game_prior = _safe_float(row.get("game_breach_rate_sd"))
    n_game_sd = _safe_float(row.get("n_game_sd"))

    # ---- build X ----
    missing_feats = [c for c in feature_cols if c not in df_row.columns]
    if missing_feats:
        raise ValueError(f"Row is missing {len(missing_feats)} feature_cols, e.g. {missing_feats[:10]}")
    X = df_row[feature_cols]

    # ---- predict ----
    p_model = float(model.predict_proba(X)[:, 1][0])
    p_adjusted, blend_weight = _compute_adjusted_probability(
        p_model=p_model,
        game_breach_rate_sd=game_prior,
        n_game_sd=n_game_sd,
        shrinkage=20.0,
    )
    margin = float(p_adjusted - thr)

    # ---- label if present ----
    actual = None
    for c in label_cols:
        if c in df_row.columns:
            actual = df_row.iloc[0][c]
            break
    actual_label = int(actual) if actual is not None and pd.notna(actual) else None

    # ---- historical context (per-game, pre-split, same snapshot_day) ----
    ref_pool = _build_reference_pool(
        df_all=df_all,
        split_month=split_month,
        snapshot_day=snapshot_day,
        game_id=game_id,
        feature_cols=feature_cols,
        month_col=MONTH_COL,
    )
    if "roas_breach" in ref_pool.columns and len(ref_pool) > 0:
        base_rate = float(ref_pool["roas_breach"].astype(int).mean())
        n_hist = int(len(ref_pool))
    else:
        base_rate, n_hist = None, 0

    historic_context = {
    "baseline_breach_rate_same_snapshot_day_pre_split": base_rate,
    "n_history_rows_snapshot_day_pre_split": n_hist,
    "baseline_breach_rate_game_snapshot_day": game_prior,
    "n_game_snapshot_day_pre_split": n_game_sd,
    "adjustment_blend_weight": blend_weight,
    }

    # ---- client metrics pack ----
    metrics = _compute_client_metrics(row)

    # ---- SHAP drivers ----
    whitelist = driver_whitelist or CLIENT_DRIVER_WHITELIST
    whitelist = [f for f in whitelist if f in feature_cols]  # only if present in model
    # TreeExplainer (binary): ensure we take class-1 shap values when list is returned
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[-1])[0]
    else:
        shap_row = np.array(shap_values)[0]

    vals_row = X.iloc[0].to_numpy()
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "value_raw": vals_row,
        "shap": shap_row,
    })
    shap_df["abs_shap"] = shap_df["shap"].abs()
    shap_df["direction"] = np.where(shap_df["shap"] >= 0, "increases breach risk", "decreases breach risk")

    # filter to client-safe drivers + sort by abs impact
    shap_df = shap_df[shap_df["feature"].isin(whitelist)].copy()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    # percentiles vs per-game historical pool (pre-split)
    # (same snapshot_day already enforced in ref_pool)
    percentiles = []
    p10s, p50s, p90s = [], [], []
    for f, v in zip(shap_df["feature"].tolist(), shap_df["value_raw"].tolist()):
        if n_hist >= 20 and f in ref_pool.columns:
            ref = ref_pool[f].to_numpy()
            v_float = _safe_float(v)
            percentiles.append(_percentile_of_value(v_float, ref))
            ref_clean = np.asarray(ref)
            ref_clean = ref_clean[np.isfinite(ref_clean)]
            if ref_clean.size >= 20:
                p10s.append(float(np.quantile(ref_clean, 0.10)))
                p50s.append(float(np.quantile(ref_clean, 0.50)))
                p90s.append(float(np.quantile(ref_clean, 0.90)))
            else:
                p10s.append(None); p50s.append(None); p90s.append(None)
        else:
            percentiles.append(None)
            p10s.append(None); p50s.append(None); p90s.append(None)

    shap_df["value"] = shap_df.apply(lambda r: _format_value(r["feature"], r["value_raw"]), axis=1)
    shap_df["label"] = shap_df["feature"].map(_feature_label)
    shap_df["unit"] = shap_df["feature"].map(_feature_unit)
    shap_df["percentile_same_game_pre_split"] = percentiles
    shap_df["p10_same_game_pre_split"] = p10s
    shap_df["p50_same_game_pre_split"] = p50s
    shap_df["p90_same_game_pre_split"] = p90s

    top_k = shap_df.head(int(k)).copy()


    # --- historically adjusted probabilities ---
    ref_prob_adj = np.array([])

    if len(ref_pool) > 0:
        p_ref_model = model.predict_proba(ref_pool[feature_cols])[:, 1]

    if "game_breach_rate_sd" in ref_pool.columns:
        gp_ref = ref_pool["game_breach_rate_sd"].astype(float).to_numpy()
    else:
        gp_ref = np.full(len(ref_pool), np.nan)

    if "n_game_sd" in ref_pool.columns:
        n_ref = ref_pool["n_game_sd"].astype(float).to_numpy()
    else:
        n_ref = np.full(len(ref_pool), np.nan)

    ref_prob_adj = np.array([
        _compute_adjusted_probability(
            p_model=pm,
            game_breach_rate_sd=(None if np.isnan(gp) else gp),
            n_game_sd=(None if np.isnan(nr) else nr),
            shrinkage=10.0,
        )[0]
        for pm, gp, nr in zip(p_ref_model, gp_ref, n_ref)
    ])

    risk_percentile = _risk_percentile(p_adjusted, ref_prob_adj)
    if ref_prob_adj.size >= 20:
        bucket_info = _risk_bucket_from_quantiles(p_adjusted, ref_prob_adj)
    else:
        bucket_info = _risk_bucket_from_margin(p_adjusted, thr)


    # ---- actions ----
    actions = _recommend_actions(p_adjusted, thr, metrics, top_k)

    # ---- assemble payload ----
    meta = {
        "game_id": game_id,
        "snapshot_day": snapshot_day,
        COHORT_COL: cohort_ts,
    }
    if month_ts is not None:
        meta[MONTH_COL] = month_ts
    if actual_label is not None:
        meta["actual_label"] = actual_label

    headline = {
        "p_breach_model": p_model,
        "p_breach_adjusted": p_adjusted,
        "threshold": thr,
        "margin_adjusted": margin,
        "risk_percentile_snapshot_day": risk_percentile,
        "risk_bucket": bucket_info["bucket"],
        }

    return {
        "meta": meta,
        "headline": headline,
        "historic_context": historic_context,
        "metrics": metrics,
        "drivers_top_k": top_k[[
            "feature", "label", "unit", "value", "direction",
            "shap", "abs_shap",
            "percentile_same_game_pre_split",
            "p10_same_game_pre_split", "p50_same_game_pre_split", "p90_same_game_pre_split",
        ]].copy(),
        "drivers_all": shap_df,  # keep for internal/debug; you can drop when serving clients
        "actions": actions,
    }


