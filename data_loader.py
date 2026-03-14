from google.cloud import bigquery
import pandas as pd
from config import TABLE_DAILY, TABLE_COHORT, TABLE_GAME_ID, PREDICT_MONTH
from datetime import datetime
import calendar

dt = datetime.strptime(PREDICT_MONTH, "%Y-%m-%d")
last_day = calendar.monthrange(dt.year, dt.month)[1]
CUTOFF_DATE = f"{dt.year}-{dt.month:02d}-{last_day:02d}"



def load_daily_data(client: bigquery.Client) -> pd.DataFrame:

    QUERY = f"""
    SELECT
      source_system,
      client_id,
      game_id,
      cohort,
      month,
      d,
      rev_d,
      rev_7d,
      rev_14d,
      spend_d,
      spend_7d,
      spend_14d,
      gamma_eom
    FROM `{TABLE_DAILY}`
    """

    df = client.query(QUERY).to_dataframe()

    df["d"] = pd.to_datetime(df["d"])
    df["month"] = pd.to_datetime(df["month"])
    
    df = df[df["d"] <= CUTOFF_DATE].copy()
    
    df = df[df["rev_7d"] >= 0].copy()

    grain = ["source_system","client_id","game_id","cohort","month","d"]
    dups = df.duplicated(subset=grain).sum()

    if dups > 0:
        raise ValueError(f"Daily grain violated: {dups}")

    return df


def load_cohort_features(client: bigquery.Client) -> pd.DataFrame:

    q = f"""
    SELECT
      source_system,
      client_id,
      game_id,
      month,
      snapshot_day,

      cohort_spend_weight_sum,
      n_cohorts_included,
      spend_share_last_3_cohorts,
      cohort_age_wmean,

      rpi_d1_wmean,
      rpi_d3_wmean,
      rpi_d7_wmean,
      ret_d1_wmean,
      ret_d3_wmean,
      ret_d7_wmean,
      rpi_ratio_d3_d1,
      rpi_ratio_d7_d3
    FROM `{TABLE_COHORT}`
    """

    df = client.query(q).to_dataframe()

    df["month"] = pd.to_datetime(df["month"])
    df["snapshot_day"] = df["snapshot_day"].astype(int)

    df = df[df["month"] <= CUTOFF_DATE].copy()


    return df



def load_game_ids(client: bigquery.Client) -> list[str]:
    q = f"""
    SELECT game_id
    FROM `{TABLE_GAME_ID}`
    """

    df = client.query(q).to_dataframe()

    return df["game_id"].tolist()
