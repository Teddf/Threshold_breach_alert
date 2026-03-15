import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
    precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import os

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import *
from model import train_roas_breach_model



# -------------------
# time-aware split
# -------------------

def time_aware_split_by_month(df_labeled: pd.DataFrame, split_month: str = SPLIT_MONTH):
    """
    Time-aware split by month cutoff.
    Assumes df_labeled['month'] is datetime64[ns] at month-start.
    """
    split_ts = pd.Timestamp(split_month)
    train = df_labeled[df_labeled["month"] < split_ts].copy()
    test  = df_labeled[df_labeled["month"] >= split_ts].copy()
    return train, test

# -------------------
# find best threshold
# -------------------
def find_best_threshold(
    clf,
    df: pd.DataFrame,
    feature_cols: list[str],
    metric: str = "recall",          # "precision" | "f1" | "recall"
    t_min: float = 0.05,
    t_max: float = 0.95,
    n_grid: int = 91,
    min_pred_pos: int = 1,              # avoid “precision=1.0” with 0 predicted positives
):
    X = df[feature_cols]
    y = df["roas_breach"].astype(int).to_numpy()
    p = clf.predict_proba(X)[:, 1]

    thresholds = np.linspace(t_min, t_max, n_grid)

    best_t = None
    best_score = -1.0

    for t in thresholds:
        y_hat = (p >= t).astype(int)
        pred_pos = int(y_hat.sum())
        if pred_pos < min_pred_pos:
            continue

        tp = int(((y_hat == 1) & (y == 1)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        fn = int(((y_hat == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        elif metric == "f1":
            score = f1
        else:
            raise ValueError("metric must be one of: 'precision', 'recall', 'f1'")

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def get_day_models(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    split_month: str,
    snapshot_days=(7, 14, 21, 28),
    model_params=None,
    use_scale_pos_weight=True,
):
    '''
    returns trained models per snapshot day.
    '''
    models_by_sd = {}

    for sd in snapshot_days:
        df_sd = df_all[df_all["snapshot_day"] == sd].copy()
        if df_sd.empty:
            continue

        # time split
        train_df, test_df = time_aware_split_by_month(df_sd, split_month=split_month)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        if train_df["roas_breach"].nunique() < 2 or test_df["roas_breach"].nunique() < 2:
            continue

        # ----- fit base model -----
        clf = train_roas_breach_model(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=None,
            use_scale_pos_weight=use_scale_pos_weight,
        )
        models_by_sd[sd] = clf
    return models_by_sd


def find_best_thresholds_by_snapshot_day(
    models_by_sd: dict,
    df_all: pd.DataFrame,
    split_month: str,
    snapshot_days=(7, 14, 21, 28),
    metric: str = "recall",
    feature_cols=None,   # list[str] OR dict[int, list[str]] OR None -> use model.feature_names_in_
    **threshold_kwargs,
):
    """
    feature_cols:
      - None: use models_by_sd[sd].feature_names_in_ (recommended; avoids mismatch)
      - list[str]: same features for all snapshot_days
      - dict[int, list[str]]: per-snapshot_day features {sd: [...]}
    """
    split_ts = pd.Timestamp(split_month)

    rows = []
    best_thresholds = {}

    for sd in snapshot_days:
        if sd not in models_by_sd:
            continue

        df_test = df_all[(df_all["snapshot_day"] == sd) & (df_all["month"] >= split_ts)].copy()
        if df_test.empty or df_test["roas_breach"].nunique() < 2:
            continue

        # pick feature cols for this snapshot_day
        if feature_cols is None:
            feature_cols_sd = list(models_by_sd[sd].feature_names_in_)
        elif isinstance(feature_cols, dict):
            feature_cols_sd = list(feature_cols[sd])
        else:
            feature_cols_sd = list(feature_cols)

        # guard: ensure features exist in df_test
        missing = [c for c in feature_cols_sd if c not in df_test.columns]
        if missing:
            raise ValueError(f"Missing features for snapshot_day={sd}: {missing}")

        t, score = find_best_threshold(
            clf=models_by_sd[sd],
            df=df_test,
            feature_cols=feature_cols_sd,
            metric=metric,
            **threshold_kwargs,
        )

        best_thresholds[sd] = float(t)
        rows.append({
            "snapshot_day": int(sd),
            "n_test": int(len(df_test)),
            "pos_test": int(df_test["roas_breach"].sum()),
            f"best_{metric}": float(score),
            "best_threshold": float(t),
        })

    return pd.DataFrame(rows).sort_values("snapshot_day"), best_thresholds

# Separate models per snapshot_day + OPTIONAL isotonic calibration (fit on train-val, apply to test)
# Produces a metrics table you can paste into a report + stores models/calibrators per snapshot_day.



def _binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def _ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE with bins on predicted probability p in [0,1].
    For each bin: |avg(p) - avg(y)| weighted by bin fraction.
    """
    y_true = y_true.astype(int)
    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(p)
    if n == 0:
        return float("nan")

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)

        cnt = int(m.sum())
        if cnt == 0:
            continue

        avg_p = float(p[m].mean())
        avg_y = float(y_true[m].mean())
        ece += (cnt / n) * abs(avg_p - avg_y)

    return float(ece)


def _per_game_snapshot_precision_recall(scored_by_sd: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sd, df in scored_by_sd.items():
        if df is None or df.empty or ("game_id" not in df.columns):
            continue

        g = df.groupby("game_id", as_index=False).apply(
            lambda x: pd.Series({
                "n": int(len(x)),
                "pos_true": int(x["y_true"].sum()),
                "pos_pred": int(x["y_hat"].sum()),
                "tp": int(((x["y_true"] == 1) & (x["y_hat"] == 1)).sum()),
                "precision_mean": float(precision_score(x["y_true"], x["y_hat"], zero_division=0)),
                "recall_mean": float(recall_score(x["y_true"], x["y_hat"], zero_division=0)),
            })
        ).reset_index(drop=True)

        g.insert(0, "snapshot_day", int(sd))
        rows.append(g)

    if not rows:
        return pd.DataFrame(columns=["snapshot_day","game_id","n","pos_true","pos_pred","tp","precision_mean","recall_mean"])

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["game_id", "snapshot_day"])
        .reset_index(drop=True)
    )


def train_eval_models_by_snapshot_day(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    split_month: str,
    thresholds_by_sd: dict | None = None,  # snapshot_day -> threshold
    snapshot_days=(7, 14, 21, 28),
    model_params=None,
    use_scale_pos_weight=True,

    # NEW:
    make_plots: bool = True,
    save_dir: str | None = None,     # if set, saves PNGs here
    return_figs: bool = True,         # returns dict[snapshot_day] -> fig
    agg_by_game: bool = True,         # x-axis uses game ordering by mean p
    compute_per_game_snapshot: bool = False,
):
    """
    Returns:
      metrics_df, models_by_sd, scored_by_sd, (optional) figs_by_sd
    """
    results = []
    scored_by_sd = {}
    figs_by_sd = {}
    models_by_sd= get_day_models(
    df_all,
    feature_cols,
    split_month= split_month,
    snapshot_days=snapshot_days,
    model_params=None,
    use_scale_pos_weight=True,
    )
    if thresholds_by_sd is None:
      table, thresholds_by_sd = find_best_thresholds_by_snapshot_day(
        models_by_sd=models_by_sd,
        df_all=df_all,
        split_month=split_month,
        snapshot_days=snapshot_days,
        feature_cols=feature_cols,   # single list
        metric="f1",
        t_min=0.05,
        t_max=0.95,
        n_grid=181,
        min_pred_pos=3,
      )


    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)



    for sd in snapshot_days:
        df_sd = df_all[df_all["snapshot_day"] == sd].copy()
        if df_sd.empty:
            continue

        train_df, test_df = time_aware_split_by_month(df_sd, split_month=split_month)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        if train_df["roas_breach"].nunique() < 2 or test_df["roas_breach"].nunique() < 2:
            continue
        clf = models_by_sd[sd]

        # calibrate
        cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
        cal.fit(train_df[feature_cols], train_df["roas_breach"].astype(int).to_numpy())


        # ---- predict ----
        X_test = test_df[feature_cols]
        y_test = test_df["roas_breach"].astype(int).to_numpy()
        p = clf.predict_proba(X_test)[:, 1]

        # ---- threshold ----
        threshold = float(thresholds_by_sd.get(sd, 0.5))
        y_hat = (p >= threshold).astype(int)

        # ---- confidence (numeric) ----
        # confidence in predicted class (0.5..1.0), higher => more certain
        conf = np.maximum(p, 1 - p)
        # entropy (0..~0.693), lower => more certain
        ent = _binary_entropy(p)
        correct = (y_hat == y_test).astype(int)

        # ---- metrics ----
        auc = roc_auc_score(y_test, p)
        ap = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)

        precision = precision_score(y_test, y_hat, zero_division=0)
        recall = recall_score(y_test, y_hat, zero_division=0)
        f1 = f1_score(y_test, y_hat, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()

        ece_10 = _ece_binary(y_test, p, n_bins=10)
        mean_conf = float(np.mean(conf))
        median_conf = float(np.median(conf))
        mean_ent = float(np.mean(ent))
        median_ent = float(np.median(ent))

        conf_correct = float(np.mean(conf[correct == 1])) if np.any(correct == 1) else float("nan")
        conf_incorrect = float(np.mean(conf[correct == 0])) if np.any(correct == 0) else float("nan")

        results.append({
            "snapshot_day": int(sd),
            "threshold": float(threshold),
            "n_test": int(len(test_df)),
            "pos_test": int(y_test.sum()),

            "roc_auc": float(auc),
            "pr_auc": float(ap),
            "brier": float(brier),
            "ece_10": float(ece_10),

            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),

            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),

            # confidence summaries (how sure was the model)
            "mean_confidence": mean_conf,
            "median_confidence": median_conf,
            "mean_entropy": mean_ent,
            "median_entropy": median_ent,
            "mean_confidence_correct": conf_correct,
            "mean_confidence_incorrect": conf_incorrect,
        })

        # ---- scored rows ----
        id_cols = ["source_system", "client_id", "game_id", "cohort", "month", "snapshot_day"]
        id_cols = [c for c in id_cols if c in test_df.columns]
        out = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        out["y_true"] = y_test
        out["p_breach"] = p
        out["y_hat"] = y_hat
        out["confidence"] = conf
        out["entropy"] = ent
        out["correct"] = correct
        scored_by_sd[sd] = out

        # ---- plots ----
        if make_plots:
            if "game_id" not in out.columns:
                # no plot possible without game_id
                pass
            else:
                plot_df = out.copy()

                # order games by mean predicted probability (readability)
                game_order = (
                    plot_df.groupby("game_id", as_index=True)["p_breach"]
                    .mean()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )

                if agg_by_game:
                    # one point per game: mean p; color by majority y_true in test rows
                    g = plot_df.groupby("game_id", as_index=False).agg(
                        p_breach_mean=("p_breach", "mean"),
                        y_true_rate=("y_true", "mean"),
                        n=("y_true", "size"),
                        conf_mean=("confidence", "mean"),
                        ent_mean=("entropy", "mean"),
                    )
                    g["y_true_majority"] = (g["y_true_rate"] >= 0.5).astype(int)
                    g["game_id"] = pd.Categorical(g["game_id"], categories=game_order, ordered=True)
                    g = g.sort_values("game_id")

                    x = np.arange(len(g))
                    y = g["p_breach_mean"].to_numpy()
                    colors = np.where(g["y_true_majority"].to_numpy() == 1, "red", "green")

                    fig, ax = plt.subplots(figsize=(max(10, len(g) * 0.25), 5))
                    ax.scatter(x, y, c=colors, alpha=0.85)

                    ax.axhline(threshold, linestyle="--")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_ylabel("Predicted breach probability (mean)")
                    ax.set_xlabel("game_id (sorted by mean p)")
                    ax.set_title(f"Snapshot day {sd}: P(breach) by game_id (color = actual outcome majority)")
                    ax.set_xticks(x)
                    ax.set_xticklabels(g["game_id"].astype(str).tolist(), rotation=90, ha="center")

                    # compact annotation: overall confidence & calibration
                    ax.text(
                        0.01, 0.02,
                        f"mean_conf={mean_conf:.3f} | mean_ent={mean_ent:.3f} | ece_10={ece_10:.3f} | thr={threshold:.3f}",
                        transform=ax.transAxes
                    )

                    fig.tight_layout()

                else:
                    # one point per row/cohort: jittered categorical x
                    plot_df["game_id"] = pd.Categorical(plot_df["game_id"], categories=game_order, ordered=True)
                    plot_df = plot_df.sort_values("game_id")
                    x0 = plot_df["game_id"].cat.codes.to_numpy().astype(float)
                    jitter = (np.random.RandomState(0).randn(len(x0)) * 0.08)
                    x = x0 + jitter
                    y = plot_df["p_breach"].to_numpy()
                    colors = np.where(plot_df["y_true"].to_numpy() == 1, "red", "green")

                    fig, ax = plt.subplots(figsize=(max(10, len(game_order) * 0.25), 5))
                    ax.scatter(x, y, c=colors, alpha=0.6, s=18)

                    ax.axhline(threshold, linestyle="--")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_ylabel("Predicted breach probability")
                    ax.set_xlabel("game_id (sorted by mean p)")
                    ax.set_title(f"Snapshot day {sd}: P(breach) by game_id (color = actual outcome)")
                    ax.set_xticks(np.arange(len(game_order)))
                    ax.set_xticklabels([str(gid) for gid in game_order], rotation=90, ha="center")

                    ax.text(
                        0.01, 0.02,
                        f"mean_conf={mean_conf:.3f} | mean_ent={mean_ent:.3f} | ece_10={ece_10:.3f} | thr={threshold:.3f}",
                        transform=ax.transAxes
                    )

                    fig.tight_layout()

                if save_dir is not None:
                    path = os.path.join(save_dir, f"p_breach_by_game_snapshot_day_{sd}.png")
                    fig.savefig(path, dpi=160)

                if return_figs:
                    figs_by_sd[sd] = fig
                else:
                    plt.close(fig)

    metrics_df = pd.DataFrame(results).sort_values("snapshot_day")
    per_game_sd = None
    if compute_per_game_snapshot:
      per_game_sd = _per_game_snapshot_precision_recall(scored_by_sd)

    if return_figs:
      if compute_per_game_snapshot:
        return metrics_df, per_game_sd, models_by_sd, scored_by_sd, figs_by_sd
      return metrics_df, models_by_sd, scored_by_sd, figs_by_sd

    if compute_per_game_snapshot:
        return metrics_df, per_game_sd, models_by_sd, scored_by_sd
    return metrics_df, models_by_sd, scored_by_sd

# --------------------
# CALL 
# --------------------
'''
metrics_df, models_by_sd,scored_by_sd,figs_by_sd = train_eval_models_by_snapshot_day(
    df_all=df_model_final,
    feature_cols=feature_cols,
    split_month=SPLIT_MONTH,
    thresholds_by_sd = None,
    snapshot_days=(7,14,21,28),
    make_plots=False,
    save_dir=None,        # or "plots/"
    agg_by_game=False,     # one point per game (mean)
    compute_per_game_snapshot = False,
)


old_metrics = metrics_df[[
    "snapshot_day",
    "threshold",
    "n_test",
    "pos_test",
    "precision",
    "recall",
    "f1",
    "brier",
    "roc_auc",
    "pr_auc",
    "tp","fp","tn","fn",
    "ece_10"
]].sort_values("snapshot_day")

# per_game_sd
# metrics_df
old_metrics
'''






