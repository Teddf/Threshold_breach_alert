import pickle
import pandas as pd


from config import *
from explainability_shap_explainer import explain_breach_for_client





# -----------------------
# Load artifacts once
# -----------------------

with open("artifacts/breach_model/models_by_sd.pkl", "rb") as f:
    models_by_sd = pickle.load(f)

with open("artifacts/breach_model/thresholds_by_sd.pkl", "rb") as f:
    thresholds_by_sd = pickle.load(f)

with open("artifacts/breach_model/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)


with open ("artifacts/breach_model/df_model_final.pkl", "rb") as f:
    df_model = pickle.load(f)

# -----------------------
# Main serving function
# -----------------------

def score_and_explain(
    game_id: str,
    snapshot_day: int,
    cohort_month: str,
    df_features: pd.DataFrame = df_model,
    split_month: str = SPLIT_MONTH,
    thresholds_by_sd = thresholds_by_sd,
    models_by_sd = models_by_sd,
):
    """
    df_features must already contain the model features
    """

    # ---- explanation payload
    explanation = explain_breach_for_client(
        df_all=df_features,
        models_by_sd=models_by_sd,
        thresholds_by_sd=thresholds_by_sd,
        feature_cols=feature_cols,
        split_month=split_month,
        game_id=game_id,
        snapshot_day=snapshot_day,
        cohort_month=cohort_month,
        k=3, 
    )


    if include_explanation and serve_output_client: 
        prepare_explanation =   {
            "headline": explanation["headline"],
            "metrics": explanation["metrics"],
            "drivers_top_k": explanation["drivers_top_k"],
            "actions": explanation["actions"],
        }
    elif serve_output_client: 
        prepare_explanation =   {
            "headline": explanation["headline"],
            "metrics": explanation["metrics"],
            "drivers_top_k": explanation["drivers_top_k"],
        }
    else: 
        prepare_explanation = {"meta": explanation["meta"],
        "headline": explanation["headline"],
        "historic_context": explanation["historic_context"],            
        "metrics": explanation["metrics"],
        "drivers_top_k": explanation["drivers_top_k"],
        "actions": explanation["actions"],
        }

    return {
        "explanation": prepare_explanation,
    }