import pandas as pd
from config import SPLIT_MONTH


def build_snapshots(df, cutoff_days=[7,14,21,28], min_periods_std=2):

    keys = ["source_system","client_id","game_id","cohort","month"]

    x = df.copy()
    x = x.sort_values(keys + ["d"], kind="mergesort")

    x["snapshot_day"] = x["d"].dt.day
    x["days_in_month"] = x["d"].dt.daysinmonth
    x["days_remaining"] = x["days_in_month"] - x["snapshot_day"]

    x["R_MTD"] = x.groupby(keys)["rev_d"].cumsum()
    x["S_MTD"] = x.groupby(keys)["spend_d"].cumsum()

    x["rev_7d_std"] = (
        x.groupby(keys)["rev_d"]
        .rolling(window=7,min_periods=min_periods_std)
        .std()
        .reset_index(level=keys, drop=True)
    )

    x["spend_7d_std"] = (
        x.groupby(keys)["spend_d"]
        .rolling(window=7,min_periods=min_periods_std)
        .std()
        .reset_index(level=keys, drop=True)
    )

    snap = x[x["snapshot_day"].isin(cutoff_days)].copy()

    out_cols = keys + [
        "snapshot_day",
        "days_in_month","days_remaining",
        "R_MTD","S_MTD",
        "rev_7d","rev_14d","rev_7d_std",
        "spend_7d","spend_14d","spend_7d_std",
    ]

    snap = snap[out_cols]

    snap = snap.drop_duplicates(subset=keys + ["snapshot_day"])

    return snap


def time_aware_split_by_month(df_labeled: pd.DataFrame, split_month: str = SPLIT_MONTH):
    """
    Time-aware split by month cutoff.
    Assumes df_labeled['month'] is datetime64[ns] at month-start.
    """
    split_ts = pd.Timestamp(split_month)
    train = df_labeled[df_labeled["month"] < split_ts].copy()
    test  = df_labeled[df_labeled["month"] >= split_ts].copy()
    return train, test