import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ================================
# Load datasets
# ================================
baseline_nst_df = pd.read_csv('./results/archived/chemprop_permeability_nst_500_checkpoint/chemprop_test_performance.csv')
baseline_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_fusion_nst_checkpoint/chemprop_test_performance.csv')

mtb_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_mtb_fusion_nst_checkpoint/chemprop_test_performance.csv')
random_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_random23_fusion_nst_checkpoint/chemprop_test_performance.csv')

# ================================
# Extract RMSE columns
# ================================
baseline = baseline_nst_df["Iter0 RMSE"].values
full_fusion = baseline_fnst_df["Iter0 RMSE"].values
full_fnst = baseline_fnst_df["Iter2 RMSE"].values

mtb_fusion = mtb_fnst_df["Iter0 RMSE"].values
mtb_fnst = mtb_fnst_df["Iter2 RMSE"].values

random_fusion = random_fnst_df["Iter0 RMSE"].values
random_fnst = random_fnst_df["Iter2 RMSE"].values

# ================================
# Define comparisons
# ================================
comparisons = [
    ("Full Baseline vs. Baseline + Fusion", baseline, full_fusion),
    ("Full Baseline vs. Baseline Fusion+NST", baseline, full_fnst),
    ("MTB Baseline vs. Baseline + Fusion", baseline, mtb_fusion),
    ("MTB Baseline vs. Baseline Fusion+NST", baseline, mtb_fnst),
    ("Random23 Baseline vs. Baseline + Fusion", baseline, random_fusion),
    ("Random23 Baseline vs. Baseline Fusion+NST", baseline, random_fnst),
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
lines.append(r"\end{table}")

latex_table = "\n".join(lines)
print(latex_table)
