import pandas as pd
from sklearn.metrics import precision_score, recall_score


def per_game_snapshot_precision_recall(scored_by_sd):

    rows = []

    for sd, df in scored_by_sd.items():

        if df.empty:
            continue

        g = df.groupby("game_id").apply(
            lambda x: pd.Series({
                "n": len(x),
                "tp": ((x["y_true"] == 1) & (x["y_hat"] == 1)).sum(),
                "precision": precision_score(x["y_true"], x["y_hat"], zero_division=0),
                "recall": recall_score(x["y_true"], x["y_hat"], zero_division=0),
            })
        ).reset_index()

        g["snapshot_day"] = sd

        rows.append(g)

    return pd.concat(rows)