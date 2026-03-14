import numpy as np
import pandas as pd


def predict_month(df_model, models_by_sd, thresholds_by_sd, feature_cols):

    df_pred = df_model.copy()

    df_pred["p_breach"] = np.nan

    for sd, g in df_pred.groupby("snapshot_day"):

        if sd not in models_by_sd:
            continue

        X = g[feature_cols]

        df_pred.loc[g.index, "p_breach"] = (
            models_by_sd[sd]
            .predict_proba(X)[:,1]
        )

    df_pred["threshold"] = df_pred["snapshot_day"].map(thresholds_by_sd)

    df_pred["margin"] = df_pred["p_breach"] - df_pred["threshold"]

    return df_pred