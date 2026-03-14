import numpy as np
import pandas as pd


def outlier_error_lift_report(
    sc,
    train_df,
    feature_cols,
    q=0.01,
    min_outliers=20,
):

    rows = []

    for col in feature_cols:

        if col not in sc.columns or col not in train_df.columns:
            continue

        tr = train_df[col].dropna()

        q_low = tr.quantile(q)
        q_high = tr.quantile(1 - q)

        x = sc[col]

        is_out = ((x < q_low) | (x > q_high)).fillna(False)

        n_out = int(is_out.sum())
        n_in = int((~is_out).sum())

        if n_out < min_outliers:
            continue

        err_out = float(sc.loc[is_out, "error"].mean())
        err_in = float(sc.loc[~is_out, "error"].mean())

        rows.append({
            "feature": col,
            "error_outlier": err_out,
            "error_normal": err_in,
            "error_lift": err_out - err_in,
            "n_outliers": n_out,
        })

    return pd.DataFrame(rows).sort_values("error_lift", ascending=False)