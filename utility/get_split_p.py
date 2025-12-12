import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ================================
# Load datasets
# ================================
hard_df = pd.read_csv('./results/archived/chemprop_permeability_checkpoint/chemprop_test_performance.csv')
hard_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_fusion_nst_checkpoint/chemprop_test_performance.csv')

random_df = pd.read_csv('./results/archived/chemprop_permeability_random_checkpoint/chemprop_test_performance.csv')
random_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_random_fusion_nst_checkpoint/chemprop_test_performance.csv')

easy_df = pd.read_csv('./results/archived/chemprop_permeability_easy_checkpoint/chemprop_test_performance.csv')
easy_fnst_df = pd.read_csv('./results/archived/chemprop_permeability_easy_fusion_nst_checkpoint/chemprop_test_performance.csv')

# ================================
# Extract RMSE columns
# ================================
hard = hard_df["RMSE"].values
hard_fnst = hard_fnst_df["Iter2 RMSE"].values

random = random_df["RMSE"].values
random_fnst = random_fnst_df["Iter2 RMSE"].values

easy = easy_df["RMSE"].values
easy_fnst = easy_fnst_df["Iter3 RMSE"].values

# ================================
# Define comparisons
# ================================
comparisons = [
    ("Scaffold-exclusive", hard, hard_fnst),
    ("Scaffold-agnostic", random, random_fnst),
    ("Scaffold-inclusive", easy, easy_fnst),

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
