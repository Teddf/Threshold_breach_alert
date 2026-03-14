from datetime import datetime
from dateutil.relativedelta import relativedelta


PREDICT_MONTH = "2026-03-01"

dt = datetime.strptime(PREDICT_MONTH, "%Y-%m-%d")
back_6 = dt - relativedelta(months=6)
result = back_6.strftime("%Y-%m-01")
SPLIT_MONTH = result

SNAPSHOT_DAYS = [7,14,21,28]

mode = "client"
model = "sfm"

BQ_PROJECT = "pvx-dev"

TABLE_DAILY = "pvx-dev.pvx_tba.fact_daily_spend_revenue_spined_all_w_gamma"
TABLE_COHORT = "pvx-dev.pvx_tba.fact_snapshot_cohort_features_all"


# Inputs
add_game_breach_rate: bool = True
game_breach_smoothing: float = 30.0

track_historic_gap: bool = False
historic_gap_smoothing: float = 20.0

serve_output_client = True

include_explanation: bool = True

# This list is for game_ids the model can be ran on, without passing filters. 
# 0 end of month spend will still be filtered. 
extra_ids =[]



FEATURE_COLS_PVX = [
    "R_MTD",
    "S_MTD",

    'has_prev7',
    "rev_7d_sum",
    "rev_14d",
    'rev_prev7_sum',

    "spend_7d_sum",
    "spend_14d",
    'spend_prev7_sum',

    "gamma_eom",

    "fraction_elapsed",

    "gamma_gap_mtd",

    "rev_gap_mtd",
    "gap_per_remaining_day",
    "late_gap", "gap_pacing",

    "rev_accel_log",
    "spend_accel_log",
]


FEATURE_COLS_CLIENT = [
    "R_MTD",
    "S_MTD",
    "fraction_elapsed",
    "gamma_gap_mtd",
    "gap_per_remaining_day",
    "rev_gap_mtd",
    "expected_rev_needed",
    "roas_velocity",

    "rpi_d1_wmean",
    "rpi_d3_wmean",
    "rpi_d7_wmean",

    "ret_d1_wmean",
    "ret_d3_wmean",
    "ret_d7_wmean",

    "cohort_age_wmean",
    "spend_share_last_3_cohorts",

    "rpi_accel_d3_d1",
    "rpi_accel_d7_d3",
    "rpi_ratio_d3_d1",
    "ret_drop_d3_d1",

    "spend_accel_log",
    "cohort_capital_interaction",
    "curve_convexity",
    "cohort_roas_proxy_d7",
]

DIAG_COLS = [
        "spend_pacing_ratio_7d_to_prev7d",
        "spend_slope_last3_vs_prev3",
        "ROAS_MTD",
        "spend_7d_std_dyn", 
        "ROAS_EOM", 
        "gamma_eom",
        "expected_rev_needed", 
        "gamma_gap_mtd",
]

###################
# Set model details
###################
if mode == "client":
    FEATURE_COLS = FEATURE_COLS_CLIENT
elif mode == "pvx":
    FEATURE_COLS = FEATURE_COLS_PVX 
else:
    raise ValueError(f"Unsupported model: {mode}")

if model == "sfm":
    TABLE_GAME_ID = "pvx-dev.pvx_tba.game_ids_filtered_spend"
elif mode == "cm":
    TABLE_GAME_ID = "pvx-dev.pvx_tba.game_ids_filtered_cat_gen"
else:
    raise ValueError(f"Unsupported model: {model}")

ARTIFACT_DIR = "artifacts/breach_model"

