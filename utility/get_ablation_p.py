import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ================================
# Load datasets
# ================================
baseline_nst_df = pd.read_csv('./results/archived/chemprop_permeability_nst_500_checkpoint/chemprop_test_performance.csv')
baseline_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_fusion_nst_checkpoint/chemprop_test_performance.csv')

mine_df = pd.read_csv('./results/archived/chemprop_permeability_mine_checkpoint/chemprop_test_performance.csv')
unsup_df = pd.read_csv('./results/archived/chemprop_permeability_unsuper_baseline_checkpoint/chemprop_test_performance.csv')

xgb_df = pd.read_csv('/work/pi_annagreen_umass_edu/shiyun/MycoPermeNet-v2-pub/results/XGBoost_permeability_checkpoint/XGBoost_test_performance.csv')
rf_df = pd.read_csv('/work/pi_annagreen_umass_edu/shiyun/MycoPermeNet-v2-pub/results/RandomForest_permeability_checkpoint/RandomForest_test_performance.csv')

# ================================
# Extract RMSE columns
# ================================
baseline = baseline_nst_df["Iter0 RMSE"].values
baseline_nst = baseline_nst_df["Iter2 RMSE"].values
baseline_fusion = baseline_fnst_df["Iter0 RMSE"].values
baseline_fnst = baseline_fnst_df["Iter2 RMSE"].values

mine = mine_df["RMSE"].values
unsup = unsup_df["RMSE"].values

xgb = xgb_df["RMSE"].values
rf = rf_df["RMSE"].values


# ================================
# Define comparisons
# ================================
comparisons = [
    ("Baseline vs. Random Forest", baseline, rf),
    ("Baseline vs. XGBoost", baseline, xgb),
    ("Baseline vs. MINE-based baseline", baseline, mine),
    ("Baseline vs. Unsupervised + Baseline", baseline, unsup),

    ("Baseline vs. Baseline + Fusion", baseline, baseline_fusion),
    ("Baseline vs. Baseline + NST", baseline, baseline_nst),
    ("Baseline vs. Baseline Fusion+NST", baseline, baseline_fnst),

    ("Baseline + Fusion vs. Baseline Fusion+NST", baseline_fusion, baseline_fnst),
    ("Baseline + NST vs. Baseline Fusion+NST", baseline_nst, baseline_fnst),
]


# ================================
# Effect size: rank-biserial for Wilcoxon
# r = |(W+ - W-) / (W+ + W-)|
# ================================
def rank_biserial_effect(x, y):
    diff = x - y
    mask = diff != 0
    diff = diff[mask]
    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)
    # rank 1..n
    ranks = abs_diff.argsort().argsort() + 1

    W_pos = ranks[diff > 0].sum()
    W_neg = ranks[diff < 0].sum()
    if (W_pos + W_neg) == 0:
        return 0.0

    r = (W_pos - W_neg) / (W_pos + W_neg)
    return abs(r)


# ================================
# Compute Wilcoxon + effect size
# ================================
results = []
for name, a, b in comparisons:
    stat, p_raw = wilcoxon(a, b, alternative="greater")
    eff = rank_biserial_effect(a, b)
    results.append({
        "name": name,
        "p_raw": p_raw,
        "effect": eff
    })


# ================================
# Holm–Bonferroni correction
# ================================
p_raw_list = [r["p_raw"] for r in results]
reject, p_adj, _, _ = multipletests(p_raw_list, method="holm")

for i, r in enumerate(results):
    r["p_adj"] = p_adj[i]
    r["signif"] = "Yes" if p_adj[i] < 0.05 else "No"


# ================================
# Generate LaTeX table
# ================================
lines = []
lines.append(r"\begin{table}[h!]")
lines.append(r"    \centering")
lines.append(r"    \begin{tabular}{lccc}")
lines.append(r"        \toprule")
lines.append(r"        Comparison & Effect size $|\Delta|$ & Raw $p$ & Adjusted $p$ \\")
lines.append(r"        \midrule")

for r in results:
    lines.append(
        f"        {r['name']} & {r['effect']:.3f} & {r['p_raw']:.2e} & {r['p_adj']:.2e} \\\\"
    )

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \caption{Wilcoxon signed-rank test on RMSE between baseline and alternative models, reporting absolute effect sizes, raw p-values, and Holm-adjusted p-values.}")
lines.append(r"    \label{tab:signedrank_permeability}")
lines.append(r"\end{table}")

latex_table = "\n".join(lines)
print(latex_table)
