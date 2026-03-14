import os
import pickle
import json
import pandas as pd
from google.cloud import bigquery

from config import *
from data_loader import *
from snapshots import *
from labels import *
from features import *
from model import *
from evaluation import train_eval_models_by_snapshot_day


os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_pickle(obj, filename):
    with open(os.path.join(ARTIFACT_DIR, filename), "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, filename):
    with open(os.path.join(ARTIFACT_DIR, filename), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def train():
    client = bigquery.Client(project=BQ_PROJECT)

    df = load_daily_data(client)
    cohort_feat = load_cohort_features(client)
    game_filter = load_game_ids(client) + extra_ids

    df = df[df["game_id"].isin(game_filter)].copy()
    cohort_feat =  cohort_feat[cohort_feat["game_id"].isin(game_filter)].copy()


    gamma_map = (
        df[["game_id", "gamma_eom"]]
        .drop_duplicates()
        .set_index("game_id")["gamma_eom"]
    )

    snap = build_snapshots(df, SNAPSHOT_DAYS)
    dyn = compute_dyn_features_from_daily(df)
    labels = build_month_labels(df, gamma_map)

    snap_labeled = attach_labels_to_snapshots(snap, labels)

    snap_labeled = snap_labeled.merge(
        dyn,
        on=["source_system", "client_id", "game_id", "cohort", "month", "snapshot_day"],
        how="left"
    )

    snap_labeled = snap_labeled.merge(
        cohort_feat,
        on=["source_system", "client_id", "game_id", "month", "snapshot_day"],
        how="left"
    )

    df_model = add_spend_range_features(
        snap_labeled,
        S_EOM_min=50000,
        S_EOM_max=150000
    )

    if add_game_breach_rate:
        df_model = add_game_breach_rate_sd(
            df_model,
            split_month=SPLIT_MONTH,
            smoothing = game_breach_smoothing
            )
        feature_cols_final = FEATURE_COLS + ["game_breach_rate_sd"]

    if track_historic_gap: 
        df_model = add_gamma_gap_relative_to_game_history(
            df_all=df_model,
            split_month=SPLIT_MONTH,
            baseline_stat="mean",
            smoothing_n=10.0,
            out_cols_prefix="gamma_gap_rel",
        )

        # add one (or two) of these features:
        new_feats = ["gamma_gap_rel_centered"]   # start with centered later "gamma_gap_rel_z"
        feature_cols_final = feature_cols_final + new_feats
    

    keep_cols = [
        "game_id",
        "source_system",
        "client_id",
        "cohort",
        "month",
        "snapshot_day",
        "roas_breach",
    ] + feature_cols_final + DIAG_COLS

    df_model_final = df_model[keep_cols].copy()
    df_model_final = df_model_final.loc[:, ~df_model_final.columns.duplicated()]

    df_model_final, feature_cols_final = sanitize_features(df_model_final, feature_cols_final)


    metrics, models,scored, figs = train_eval_models_by_snapshot_day(
        df_all=df_model_final,
        feature_cols=feature_cols_final,
        split_month=SPLIT_MONTH,
        thresholds_by_sd = None,
        snapshot_days=(7,14,21,28),
        make_plots=False,
        save_dir=None,        # or "plots/"
        agg_by_game=False,     # one point per game (mean)
        compute_per_game_snapshot = False,
    )

    thresholds_by_sd = metrics.set_index("snapshot_day")["threshold"].to_dict()

    save_pickle(models, os.path.join(ARTIFACT_DIR, "models_by_sd.pkl"))
    save_pickle(thresholds_by_sd, os.path.join(ARTIFACT_DIR, "thresholds_by_sd.pkl"))
    save_pickle(feature_cols_final, os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))
    save_pickle(df_model_final, os.path.join(ARTIFACT_DIR, "df_model_final.pkl"))
    save_pickle(scored, os.path.join(ARTIFACT_DIR, "scored_by_sd.pkl"))

    metrics.to_csv(os.path.join(ARTIFACT_DIR, "metrics_by_sd.csv"), index=False)

    save_json(
        {
            "split_month": SPLIT_MONTH,
            "predict_month": PREDICT_MONTH,
            "snapshot_days": SNAPSHOT_DAYS,
            "mode": mode,
            "model": model,
        },
        "train_config.json"
    )

    return metrics


if __name__ == "__main__":
    metrics = train()
    print(metrics)