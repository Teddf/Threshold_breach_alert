import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_xgb_importance(models_by_sd: dict, importance_type: str = "gain") -> pd.DataFrame:
    rows = []

    for sd, model in models_by_sd.items():
        if model is None:
            continue

        if hasattr(model, "steps"):
            model = model.steps[-1][1]

        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        else:
            booster = model

        score = booster.get_score(importance_type=importance_type)

        for feat, imp in score.items():
            rows.append(
                {"snapshot_day": int(sd), "feature": feat, "importance": float(imp)}
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["importance_norm"] = (
        df["importance"]
        / df.groupby("snapshot_day")["importance"].transform("sum")
    )

    return df.sort_values(
        ["snapshot_day", "importance_norm"],
        ascending=[True, False],
    )


def permutation_importance_for_sd(
    clf,
    df_test,
    feature_cols,
    y_col="roas_breach",
    n_repeats=5,
):

    X = df_test[feature_cols]
    y = df_test[y_col].astype(int)

    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=n_repeats,
        scoring="f1",
        random_state=42,
    )

    return pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)