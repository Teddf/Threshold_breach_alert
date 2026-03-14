from xgboost import XGBClassifier
import numpy as np
import pandas as pd

def train_roas_breach_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict | None = None,
    use_scale_pos_weight: bool = True,
):
    """
Train an XGBoost binary classifier for M1 ROAS breach prediction.

Parameters
----------
train_df : pd.DataFrame
    Training dataset already split by time. Must contain:
      - 'roas_breach' (0/1 target)
      - all feature_cols
    Expected grain:
      (source_system, client_id, game_id, cohort, month, snapshot_day)

feature_cols : list[str]
    List of numeric feature column names used for training.

model_params : dict | None
    XGBClassifier parameters. If None, defaults are used.

use_scale_pos_weight : bool
    If True, automatically sets scale_pos_weight based on
    class imbalance in the training split.

Returns
-------
clf : XGBClassifier
    Fitted model trained on train_df.

Notes
-----
- Assumes time-aware split already performed externally.
- Does not perform validation or hyperparameter tuning.
- Fails if only one class is present in training data.
    """

    # y
    y_train = train_df["roas_breach"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Training data has only one class.")

    # X
    X_train = train_df[feature_cols]
    if X_train.isna().any().any():
        bad = list(X_train.columns[X_train.isna().any()])[:20]
        raise ValueError(f"NaNs in feature columns (sample): {bad}")
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Non-finite values (inf/-inf) found in features.")

    def_model_params = dict(
        n_estimators=500,        # ↓ from 800
        max_depth=3,             # ↓ from 5
        learning_rate=0.03,      # ↓ from 0.05 (compensates fewer trees)
        subsample=0.8,           # ↓ from 0.9
        colsample_bytree=0.8,    # ↓ from 0.9
        reg_lambda=5.0,          # ↑ from 1.0
        min_child_weight=5.0,    # ↑ from 1.0
        gamma=1.0,               # NEW split penalty
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    '''
    def_model_params = dict(
        n_estimators=800,        # ↓ from 800
        max_depth=5,             # ↓ from 5
        learning_rate=0.05,      # ↓ from 0.05 (compensates fewer trees)
        subsample=0.9,           # ↓ from 0.9
        colsample_bytree=0.9,    # ↓ from 0.9
        reg_lambda=1.0,          # ↑ from 1.0
        min_child_weight=1.0,    # ↑ from 1.0
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    '''
    xgb_params = dict(def_model_params)
    if model_params:
        xgb_params.update(dict(model_params))


    # imbalance
    if use_scale_pos_weight and "scale_pos_weight" not in xgb_params:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos == 0:
            raise ValueError("No positive samples in training data.")
        xgb_params["scale_pos_weight"] = neg / max(1, pos)

    clf = XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)
    return clf


