# 1. diagnosing false negatives
import numpy as np
import pandas as pd

GAME = "GS332"
SD = 28

# -----------------------------
# 0) Slice scored rows (GS338, sd=14) into FN vs TP
# -----------------------------
sc_sd_all = err_dyn[
    (err_dyn["snapshot_day"] == SD)
].copy()

sc_sd = sc_sd_all[
    (err_dyn["game_id"] == GAME)
].copy()

fn = sc_sd[(sc_sd["y_hat"] == 0) & (sc_sd["y_true"] == 1)].copy()   # false negatives
tp = sc_sd[(sc_sd["y_hat"] == 1) & (sc_sd["y_true"] == 1)].copy()   # true positives
breach_all = sc_sd_all[sc_sd_all["y_true"] == 1].copy() # all positives
no_breach_all = sc_sd_all[sc_sd_all["y_true"] == 0].copy() #  all negatives

# print("Counts:", {"FN": len(fn), "TP": len(tp)})

# -----------------------------
# 1) Merge in required raw columns from df_model_final
#    (ROAS_EOM, gamma_eom, plus day-14 features you want to compare)
# -----------------------------
keys = ["source_system", "client_id", "game_id", "cohort", "month", "snapshot_day"]
keys = [k for k in keys if (k in sc_sd.columns) and (k in df_model_final.columns)]

cols_needed = [
    "ROAS_EOM", "gamma_eom",              # step 1 (severity)
    "ROAS_MTD", "late_gap",           # step 2 (day-14 signal)
    "gap_per_remaining_day", "expected_rev_needed",
    "R_MTD", "S_MTD",
    "rev_7d_sum", "rev_prev7_sum",
    "spend_7d_sum", "spend_prev7_sum",
    "gap_pacing", "rev_accel_log", "spend_accel_log",
    "fraction_elapsed",
]
cols_needed = [c for c in cols_needed if c in df_model_final.columns]

base = df_model_final.copy()

# -----------------------------
# STEP 1) Breach severity: ROAS_EOM vs gamma_eom
# -----------------------------
for d in (fn, tp, breach_all, no_breach_all):
    d["breach_distance"] = d["ROAS_EOM"] - d["gamma_eom"]  # <0 => breach
    d["breach_margin_abs"] = d["breach_distance"].abs()

def summarize_severity(d: pd.DataFrame, label: str) -> pd.Series:
    s = d["breach_distance"].dropna()
    return pd.Series({
        "n": len(d),
        "n_with_roas_eom": int(d["ROAS_EOM"].notna().sum()),
        "mean_breach_distance": float(s.mean()) if len(s) else np.nan,
        "median_breach_distance": float(s.median()) if len(s) else np.nan,
        "p10": float(s.quantile(0.10)) if len(s) else np.nan,
        "p25": float(s.quantile(0.25)) if len(s) else np.nan,
        "p75": float(s.quantile(0.75)) if len(s) else np.nan,
        "p90": float(s.quantile(0.90)) if len(s) else np.nan,
    }, name=label)

severity_summary = pd.concat([
    summarize_severity(fn, "FN"),
    summarize_severity(tp, "TP"),
    summarize_severity(breach_all, "all pos"),
        summarize_severity(no_breach_all, "all neg")
], axis=1)

# print("\nSTEP 1 — Breach severity summary (ROAS_EOM - gamma_eom):")
# print(severity_summary)

# optional: quick “how many marginal breaches?”
MARGINAL = 0.02
'''
print("\nMarginal breach rate (|ROAS_EOM - gamma_eom| <= 0.02):")
print({
    "FN": float((fn["breach_margin_abs"] <= MARGINAL).mean()),
    "TP": float((tp["breach_margin_abs"] <= MARGINAL).mean()),
})
'''
# -----------------------------
# STEP 2) Day-14 signal: compare feature means FN vs TP
# -----------------------------
compare_features = [
    "ROAS_MTD", "late_gap", "gap_per_remaining_day", "expected_rev_needed",
    "R_MTD", "S_MTD",
    "rev_7d_sum", "rev_prev7_sum",
    "spend_7d_sum", "spend_prev7_sum",
    "gap_pacing", "rev_accel_log", "spend_accel_log",
    "fraction_elapsed",
]
compare_features = [c for c in compare_features if c in fn.columns and c in tp.columns]

step2 = pd.DataFrame({
    "FN_mean": fn[compare_features].mean(numeric_only=True),
    "TP_mean": tp[compare_features].mean(numeric_only=True),
})
step2["FN_minus_TP"] = step2["FN_mean"] - step2["TP_mean"]
step2 = step2.sort_values("FN_minus_TP", ascending=False)

'''
print("\nSTEP 2 — Day-14 feature means (FN vs TP) and difference:")
print(step2)
'''

# optional: also compare medians (more robust)
step2_med = pd.DataFrame({
    "FN_median": fn[compare_features].median(numeric_only=True),
    "TP_median": tp[compare_features].median(numeric_only=True),
})
step2_med["FN_minus_TP"] = step2_med["FN_median"] - step2_med["TP_median"]
step2_med = step2_med.sort_values("FN_minus_TP", ascending=False)

# print("\nSTEP 2b — Day-14 feature medians (FN vs TP) and difference:")
# print(step2_med)



# -----------------------------
# STEP 3) SHAP
# -----------------------------
model = models_by_sd[SD]
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(df_model_final[feature_cols])

shap_df = pd.DataFrame(shap_vals, columns=feature_cols)

# mean negative contributions
neg_shap = shap_df.mean().sort_values()
print(neg_shap.head(15))   # strongest downward drivers




# -----------------------------
# STEP 4) trajectory
# -----------------------------
gs_fn =fn


# track same cohorts over snapshot days
fn_full = err_dyn[
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 1)
].copy()

no_breach_all_full = err_dyn[
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 0)
].copy()




traj = fn_full.copy()
traj.sort_values(["cohort","month","snapshot_day"])

traj2 = no_breach_all_full.copy()
traj2.sort_values(["cohort","month","snapshot_day"])
'''
print(traj[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))


print(traj2[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))
'''

# -----------------------------
# STEP 5) Breach severity
# -----------------------------
gs_fn["breach_distance"] = gs_fn["ROAS_EOM"] - gs_fn["gamma_eom"]
no_breach_all["breach_distance"] = no_breach_all["ROAS_EOM"] - no_breach_all["gamma_eom"]




mean_severity_fn = gs_fn["breach_distance"].mean()
prob_mean_fn = gs_fn["p_breach"].mean()

mean_severity_an = no_breach_all["breach_distance"].mean()
prob_mean_an = no_breach_all["p_breach"].mean()

'''
print("Mean breach_distance for false negatives:", mean_severity_fn)
print("Mean predicted prob for false negatives:", prob_mean_fn)
print("Mean breach_distance for all negatives (all games):", mean_severity_an)
print("Mean predicted prob for all negatives (all games):", prob_mean_an)


# crude classification
if abs(mean_severity) < 0.02:
    print("Bucket: Marginal breaches (acceptable ambiguity)")
elif prob_mean < 0.40:
    print("Bucket: Structural suppression (model blind)")
else:
    print("Bucket: Likely temporal information delay")
'''

# STEP 1 — Compare GS338 FN vs All Other True Positives

# GS338 false negatives
gs_fn = err_dyn[
    (err_dyn["snapshot_day"] == SD) &
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 1)
].copy()

# All other true positives at same snapshot
other_tp = err_dyn[
    (err_dyn["snapshot_day"] == SD) &
    (err_dyn["game_id"] != GAME) &
    (err_dyn["y_true"] == 1) &
    (err_dyn["y_hat"] == 1)
].copy()

# print(len(gs_fn), len(other_tp))


# STEP 2 — Check Probability Distribution
# gs_fn["p_breach"].describe()

# STEP 3 - check base rate
breach_rate_by_game = (
    df_model_final[df_model_final["snapshot_day"] == SD]
    .groupby("game_id")["roas_breach"]
    .mean()
    .sort_values()
)

# breach_rate_by_game.loc["GS338"]



# Calls:
# step 1
'''
print("\nSTEP 1 — Breach severity summary (ROAS_EOM - gamma_eom):")
print(severity_summary)


# step 2
print("\nSTEP 2 — Day-14 feature means (FN vs TP) and difference:")
print(step2)


print("\nSTEP 2b — Day-14 feature medians (FN vs TP) and difference:")
print(step2_med)


# step 3
print("\nSTEP 3 — SHAP scores:")

neg_shap = shap_df.mean().sort_values()
print(neg_shap.head(15))

# step 4

print(traj[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))


print(traj2[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))
'''


df28 = df_model_final[df_model_final["snapshot_day"] == 28]

import matplotlib.pyplot as plt

# plt.hist(df28[df28["roas_breach"]==1]["gamma_gap_mtd"], bins=50, alpha=0.5)
plt.hist(df28[df28["roas_breach"]==0]["gamma_gap_mtd"], bins=50, alpha=0.5)
plt.show()
plt.savefig("breach_prob_snapshot28.png", dpi=300, bbox_inches="tight")
# save output


from sklearn.metrics import roc_auc_score

mask = err_dyn["game_id"] == "GS332"
roc_auc_score(err_dyn[mask]["y_true"], err_dyn[mask]["p_breach"])


# 1. diagnosing false negatives
import numpy as np
import pandas as pd

GAME = "GS332"
SD = 28

# -----------------------------
# 0) Slice scored rows (GS338, sd=14) into FN vs TP
# -----------------------------
sc_sd_all = err_dyn[
    (err_dyn["snapshot_day"] == SD)
].copy()

sc_sd = sc_sd_all[
    (err_dyn["game_id"] == GAME)
].copy()

fn = sc_sd[(sc_sd["y_hat"] == 0) & (sc_sd["y_true"] == 1)].copy()   # false negatives
tp = sc_sd[(sc_sd["y_hat"] == 1) & (sc_sd["y_true"] == 1)].copy()   # true positives
breach_all = sc_sd_all[sc_sd_all["y_true"] == 1].copy() # all positives
no_breach_all = sc_sd_all[sc_sd_all["y_true"] == 0].copy() #  all negatives

# print("Counts:", {"FN": len(fn), "TP": len(tp)})

# -----------------------------
# 1) Merge in required raw columns from df_model_final
#    (ROAS_EOM, gamma_eom, plus day-14 features you want to compare)
# -----------------------------
keys = ["source_system", "client_id", "game_id", "cohort", "month", "snapshot_day"]
keys = [k for k in keys if (k in sc_sd.columns) and (k in df_model_final.columns)]

cols_needed = [
    "ROAS_EOM", "gamma_eom",              # step 1 (severity)
    "ROAS_MTD", "late_gap",           # step 2 (day-14 signal)
    "gap_per_remaining_day", "expected_rev_needed",
    "R_MTD", "S_MTD",
    "rev_7d_sum", "rev_prev7_sum",
    "spend_7d_sum", "spend_prev7_sum",
    "gap_pacing", "rev_accel_log", "spend_accel_log",
    "fraction_elapsed",
]
cols_needed = [c for c in cols_needed if c in df_model_final.columns]

base = df_model_final.copy()

# -----------------------------
# STEP 1) Breach severity: ROAS_EOM vs gamma_eom
# -----------------------------
for d in (fn, tp, breach_all, no_breach_all):
    d["breach_distance"] = d["ROAS_EOM"] - d["gamma_eom"]  # <0 => breach
    d["breach_margin_abs"] = d["breach_distance"].abs()

def summarize_severity(d: pd.DataFrame, label: str) -> pd.Series:
    s = d["breach_distance"].dropna()
    return pd.Series({
        "n": len(d),
        "n_with_roas_eom": int(d["ROAS_EOM"].notna().sum()),
        "mean_breach_distance": float(s.mean()) if len(s) else np.nan,
        "median_breach_distance": float(s.median()) if len(s) else np.nan,
        "p10": float(s.quantile(0.10)) if len(s) else np.nan,
        "p25": float(s.quantile(0.25)) if len(s) else np.nan,
        "p75": float(s.quantile(0.75)) if len(s) else np.nan,
        "p90": float(s.quantile(0.90)) if len(s) else np.nan,
    }, name=label)

severity_summary = pd.concat([
    summarize_severity(fn, "FN"),
    summarize_severity(tp, "TP"),
    summarize_severity(breach_all, "all pos"),
        summarize_severity(no_breach_all, "all neg")
], axis=1)

# print("\nSTEP 1 — Breach severity summary (ROAS_EOM - gamma_eom):")
# print(severity_summary)

# optional: quick “how many marginal breaches?”
MARGINAL = 0.02
'''
print("\nMarginal breach rate (|ROAS_EOM - gamma_eom| <= 0.02):")
print({
    "FN": float((fn["breach_margin_abs"] <= MARGINAL).mean()),
    "TP": float((tp["breach_margin_abs"] <= MARGINAL).mean()),
})
'''
# -----------------------------
# STEP 2) Day-14 signal: compare feature means FN vs TP
# -----------------------------
compare_features = [
    "ROAS_MTD", "late_gap", "gap_per_remaining_day", "expected_rev_needed",
    "R_MTD", "S_MTD",
    "rev_7d_sum", "rev_prev7_sum",
    "spend_7d_sum", "spend_prev7_sum",
    "gap_pacing", "rev_accel_log", "spend_accel_log",
    "fraction_elapsed",
]
compare_features = [c for c in compare_features if c in fn.columns and c in tp.columns]

step2 = pd.DataFrame({
    "FN_mean": fn[compare_features].mean(numeric_only=True),
    "TP_mean": tp[compare_features].mean(numeric_only=True),
})
step2["FN_minus_TP"] = step2["FN_mean"] - step2["TP_mean"]
step2 = step2.sort_values("FN_minus_TP", ascending=False)

'''
print("\nSTEP 2 — Day-14 feature means (FN vs TP) and difference:")
print(step2)
'''

# optional: also compare medians (more robust)
step2_med = pd.DataFrame({
    "FN_median": fn[compare_features].median(numeric_only=True),
    "TP_median": tp[compare_features].median(numeric_only=True),
})
step2_med["FN_minus_TP"] = step2_med["FN_median"] - step2_med["TP_median"]
step2_med = step2_med.sort_values("FN_minus_TP", ascending=False)

# print("\nSTEP 2b — Day-14 feature medians (FN vs TP) and difference:")
# print(step2_med)



# -----------------------------
# STEP 3) SHAP
# -----------------------------
model = models_by_sd[SD]
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(df_model_final[feature_cols])

shap_df = pd.DataFrame(shap_vals, columns=feature_cols)

# mean negative contributions
neg_shap = shap_df.mean().sort_values()
print(neg_shap.head(15))   # strongest downward drivers




# -----------------------------
# STEP 4) trajectory
# -----------------------------
gs_fn =fn


# track same cohorts over snapshot days
fn_full = err_dyn[
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 1)
].copy()

no_breach_all_full = err_dyn[
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 0)
].copy()




traj = fn_full.copy()
traj.sort_values(["cohort","month","snapshot_day"])

traj2 = no_breach_all_full.copy()
traj2.sort_values(["cohort","month","snapshot_day"])
'''
print(traj[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))


print(traj2[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))
'''

# -----------------------------
# STEP 5) Breach severity
# -----------------------------
gs_fn["breach_distance"] = gs_fn["ROAS_EOM"] - gs_fn["gamma_eom"]
no_breach_all["breach_distance"] = no_breach_all["ROAS_EOM"] - no_breach_all["gamma_eom"]




mean_severity_fn = gs_fn["breach_distance"].mean()
prob_mean_fn = gs_fn["p_breach"].mean()

mean_severity_an = no_breach_all["breach_distance"].mean()
prob_mean_an = no_breach_all["p_breach"].mean()

'''
print("Mean breach_distance for false negatives:", mean_severity_fn)
print("Mean predicted prob for false negatives:", prob_mean_fn)
print("Mean breach_distance for all negatives (all games):", mean_severity_an)
print("Mean predicted prob for all negatives (all games):", prob_mean_an)


# crude classification
if abs(mean_severity) < 0.02:
    print("Bucket: Marginal breaches (acceptable ambiguity)")
elif prob_mean < 0.40:
    print("Bucket: Structural suppression (model blind)")
else:
    print("Bucket: Likely temporal information delay")
'''

# STEP 1 — Compare GS338 FN vs All Other True Positives

# GS338 false negatives
gs_fn = err_dyn[
    (err_dyn["snapshot_day"] == SD) &
    (err_dyn["game_id"] == GAME) &
    (err_dyn["y_true"] == 1)
].copy()

# All other true positives at same snapshot
other_tp = err_dyn[
    (err_dyn["snapshot_day"] == SD) &
    (err_dyn["game_id"] != GAME) &
    (err_dyn["y_true"] == 1) &
    (err_dyn["y_hat"] == 1)
].copy()

# print(len(gs_fn), len(other_tp))


# STEP 2 — Check Probability Distribution
# gs_fn["p_breach"].describe()

# STEP 3 - check base rate
breach_rate_by_game = (
    df_model_final[df_model_final["snapshot_day"] == SD]
    .groupby("game_id")["roas_breach"]
    .mean()
    .sort_values()
)

# breach_rate_by_game.loc["GS338"]



# Calls:
# step 1
'''
print("\nSTEP 1 — Breach severity summary (ROAS_EOM - gamma_eom):")
print(severity_summary)


# step 2
print("\nSTEP 2 — Day-14 feature means (FN vs TP) and difference:")
print(step2)


print("\nSTEP 2b — Day-14 feature medians (FN vs TP) and difference:")
print(step2_med)


# step 3
print("\nSTEP 3 — SHAP scores:")

neg_shap = shap_df.mean().sort_values()
print(neg_shap.head(15))

# step 4

print(traj[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))


print(traj2[[
    "snapshot_day",
    "cohort",
    "p_breach",
    "ROAS_MTD",
    "late_gap",
    "R_MTD",
    "gamma_gap_mtd",
]].head(50))
'''


df28 = df_model_final[df_model_final["snapshot_day"] == 28]

import matplotlib.pyplot as plt

# plt.hist(df28[df28["roas_breach"]==1]["gamma_gap_mtd"], bins=50, alpha=0.5)
plt.hist(df28[df28["roas_breach"]==0]["gamma_gap_mtd"], bins=50, alpha=0.5)
plt.show()
plt.savefig("breach_prob_snapshot28.png", dpi=300, bbox_inches="tight")
# save output


from sklearn.metrics import roc_auc_score

mask = err_dyn["game_id"] == "GS332"
roc_auc_score(err_dyn[mask]["y_true"], err_dyn[mask]["p_breach"])