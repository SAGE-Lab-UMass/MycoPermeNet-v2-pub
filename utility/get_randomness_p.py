import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


allVar_df = pd.read_csv('./results/archived/chemprop_all_var_checkpoint/chemprop_all_var_performance.csv')
dataVar_df = pd.read_csv('./results/archived/chemprop_data_var_checkpoint/chemprop_data_var_performance.csv')
torchVar_df = pd.read_csv('./results/archived/chemprop_torch_var_checkpoint/chemprop_torch_var_performance.csv')

allVar_r2 = allVar_df['Test R2']
allVar_rmse = allVar_df['Test RMSE']
allVar_spearman = allVar_df['Test Spearman']
dataVar_r2 = dataVar_df['Test R2']
dataVar_rmse = dataVar_df['Test RMSE']
dataVar_spearman = dataVar_df['Test Spearman']
torchVar_r2 = torchVar_df['Test R2']
torchVar_rmse = torchVar_df['Test RMSE']
torchVar_spearman = torchVar_df['Test Spearman']

results_table = f"""
\\begin{{table}}[h!]
    \\centering
    \\begin{{tabular}}{{lccc}}
        \\toprule
        Setting & Test R$^2$ & Test RMSE & Test Spearman \\\\
        \\midrule
        Vary All   & {allVar_r2.mean():.3f}±{allVar_r2.std():.3f} & {allVar_rmse.mean():.3f}±{allVar_rmse.std():.3f} & {allVar_spearman.mean():.3f} ± {allVar_spearman.std():.3f} \\\\
        Vary Data  & {dataVar_r2.mean():.3f}±{dataVar_r2.std():.3f} & {dataVar_rmse.mean():.3f}±{dataVar_rmse.std():.3f} & {dataVar_spearman.mean():.3f} ± {dataVar_spearman.std():.3f}\\\\
        Vary Torch & {torchVar_r2.mean():.3f}±{torchVar_r2.std():.3f} & {torchVar_rmse.mean():.3f}±{torchVar_rmse.std():.3f} & {torchVar_spearman.mean():.3f} ± {torchVar_spearman.std():.3f}\\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Test performance under different randomness settings.}}
    \\label{{tab:performance_summary}}
\\end{{table}}
"""
print(results_table)


comparisons = [
    ("AllVar vs. DataVar",  allVar_r2,  dataVar_r2),
    ("AllVar vs. TorchVar", allVar_r2,  torchVar_r2),
    ("DataVar vs. TorchVar", dataVar_r2, torchVar_r2),
]

results = []
for name, v1, v2 in comparisons:
    u_stat, p_raw = mannwhitneyu(v1, v2, alternative='two-sided')
    n1, n2 = len(v1), len(v2)
    # Cliff's delta / rank-biserial effect size
    delta = 2 * u_stat / (n1 * n2) - 1
    effect_size = abs(delta)
    results.append({
        "name": name,
        "u": u_stat,
        "p_raw": p_raw,
        "effect": effect_size,
    })

p_raw = [r["p_raw"] for r in results]

# Holm correction (FWER control)
rej, adj_p, _, _ = multipletests(p_raw, method='holm')

for i, r in enumerate(results):
    r["p_adj"] = adj_p[i]
    r["signif"] = "Yes" if rej[i] else "No"

lines = []
lines.append(r"\begin{table}[h!]")
lines.append(r"    \centering")
lines.append(r"    \begin{tabular}{lccc}")
lines.append(r"        \toprule")
lines.append(r"        Comparison & Effect size $|\Delta|$ & Raw $p$ & Holm-adjusted $p$ \\")
lines.append(r"        \midrule")

for r in results:
    lines.append(
        f"        {r['name']} & {r['effect']:.3f} & {r['p_raw']:.2e} & {r['p_adj']:.2e} \\\\"
    )

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \caption{Mann--Whitney $U$ test on $R^2$ between different randomness settings, reporting absolute effect size $|\Delta|$, raw $p$-values, and Holm-adjusted $p$-values.}")
lines.append(r"    \label{tab:utest_results}")
lines.append(r"\end{table}")

utest_table = "\n".join(lines)
print(utest_table)
