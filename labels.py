import numpy as np
import pandas as pd

def build_month_labels(df_daily, gamma_map, spend_eps=1e-9):

    keys = ["source_system","client_id","game_id","cohort","month"]

    labels = (
        df_daily
        .groupby(keys,as_index=False)
        .agg(
            S_EOM=("spend_d","sum"),
            R_EOM=("rev_d","sum")
        )
    )

    labels = labels[labels["S_EOM"] > 0]

    labels["gamma_eom"] = labels["game_id"].map(gamma_map)

    labels["ROAS_EOM"] = labels["R_EOM"] / np.maximum(spend_eps, labels["S_EOM"])

    labels["roas_breach"] = (labels["ROAS_EOM"] < labels["gamma_eom"]).astype(int)

    return labels


def attach_labels_to_snapshots(snap, labels):

    keys = ["source_system","client_id","game_id","cohort","month"]

    return snap.merge(labels,on=keys,how="left",validate="many_to_one")