import pandas as pd
import numpy as np


def sanitize_features(df, feature_cols):
    feature_cols = list(feature_cols)

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    nan_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()

    if nan_cols:
        nan_indicators = df[nan_cols].isna().astype(int).add_suffix("_was_nan")
        df = pd.concat([df, nan_indicators], axis=1)
        feature_cols += nan_indicators.columns.tolist()

    df = df.fillna(0.0)
    return df, feature_cols