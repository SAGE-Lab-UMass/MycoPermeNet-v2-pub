import pandas as pd
from scipy.stats import mannwhitneyu


allVar_df = pd.read_csv('./results/chemprop_all_var_checkpoint/chemprop_all_var_performance.csv')
dataVar_df = pd.read_csv('./results/chemprop_data_var_checkpoint/chemprop_data_var_performance.csv')
torchVar_df = pd.read_csv('./results/chemprop_torch_var_checkpoint/chemprop_torch_var_performance.csv')

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


def utest_row(name1, vals1, name2, vals2):
    u_stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
    signif = "Yes" if p < 0.05 else "No"
    return f"{name1} vs. {name2} & {u_stat:.2f} & {p} & {signif} \\\\"


utest_table = f"""
\\begin{{table}}[h!]
    \\centering
    \\begin{{tabular}}{{lccc}}
        \\toprule
        Comparison & U-statistic & P-value & Significant \\\\
        \\midrule
        {utest_row("AllVar", allVar_r2, "DataVar", dataVar_r2)}
        {utest_row("AllVar", allVar_r2, "TorchVar", torchVar_r2)}
        {utest_row("DataVar", dataVar_r2, "TorchVar", torchVar_r2)}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Mann-Whitney U test results on $R^2$ between different randomness settings.}}
    \\label{{tab:utest_results}}
\\end{{table}}
"""
print(utest_table)
