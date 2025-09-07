import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon


# Load data
baseline_df = pd.read_csv('./results/chemprop_permeability_checkpoint/chemprop_test_performance.csv')
baseline_nst_df = pd.read_csv('./results/chemprop_permeability_nst_500_checkpoint/chemprop_test_performance.csv')
baseline_fnst_df = pd.read_csv('./results/chemprop_permeability_fusion_nst_checkpoint/chemprop_test_performance.csv')


def plot_r2_rmse_spearman_subplots(baseline_df, baseline_nst_df, baseline_fnst_df):

    models = ['Baseline', 'Baseline + Fusion', 'Baseline + NST', 'Baseline Fusion+NST']

    metrics = ['R2', 'RMSE', 'Spearman']
    metric_labels = ['$R^2$', 'RMSE', 'Spearman']
    metric_cols = {
        'R2': ['R2', 'Iter0 R2', 'Iter1 R2', 'Iter2 R2', 'Iter3 R2'],
        'RMSE': ['RMSE', 'Iter0 RMSE', 'Iter1 RMSE', 'Iter2 RMSE', 'Iter3 RMSE'],
        'Spearman': ['Spearman', 'Iter0 Spearman', 'Iter1 Spearman', 'Iter2 Spearman', 'Iter3 Spearman']
    }

    raw_data = {
        'R2': {
            'Baseline': baseline_df[metric_cols['R2'][0]],
            'Baseline + Fusion': baseline_fnst_df[metric_cols['R2'][1]],
            'Baseline + NST': baseline_nst_df[metric_cols['R2'][3]],
            'Baseline Fusion+NST': baseline_fnst_df[metric_cols['R2'][3]],
        },
        'RMSE': {
            'Baseline': baseline_df[metric_cols['RMSE'][0]],
            'Baseline + Fusion': baseline_fnst_df[metric_cols['RMSE'][1]],
            'Baseline + NST': baseline_nst_df[metric_cols['RMSE'][3]],
            'Baseline Fusion+NST': baseline_fnst_df[metric_cols['RMSE'][3]],
        },
        'Spearman': {
            'Baseline': baseline_df[metric_cols['Spearman'][0]],
            'Baseline + Fusion': baseline_fnst_df[metric_cols['Spearman'][1]],
            'Baseline + NST': baseline_nst_df[metric_cols['Spearman'][3]],
            'Baseline Fusion+NST': baseline_fnst_df[metric_cols['Spearman'][3]],
        }
    }

    means = {m: [np.mean(raw_data[m][model]) for model in models] for m in metrics}
    stds = {m: [np.std(raw_data[m][model], ddof=1) for model in models] for m in metrics}

    # significance labels
    p_value_labels = [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]

    def get_sig_label(p):
        for thr, lab in p_value_labels:
            if p < thr:
                return lab
        return "ns"

    fig, axes = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
    model_colors = sns.color_palette("tab10", n_colors=len(models))

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        x = np.arange(len(models))

        # draw errorbar (mean ± std)
        for j in range(len(models)):
            ax.errorbar(x[j], means[metric][j], yerr=stds[metric][j],
                        fmt='o', color=model_colors[j], alpha=0.8,
                        capsize=4, elinewidth=2)

        # wilcoxon test baseline vs others
        baseline_vals = raw_data[metric]['Baseline']
        order = ['Baseline + Fusion', 'Baseline + NST', 'Baseline Fusion+NST']  # bottom → top
        # The highest point among all models' mean + std
        y_max = max(means[metric][i] + stds[metric][i] for i in range(len(models)))
        h = 0.002
        increment = 0.006

        for k, model in enumerate(order):
            vals_variant = raw_data[metric][model]
            alternative = "less" if metric in ["R2", "Spearman"] else "greater"
            try:
                stat, pval = wilcoxon(baseline_vals, vals_variant, alternative=alternative)
            except ValueError:
                pval = 1.0
            label = get_sig_label(pval)

            j = models.index(model)

            base_y = y_max + k * increment

            x1, x2 = 0, j
            ax.plot([x1, x1, x2, x2],
                    [base_y, base_y + h, base_y + h, base_y], lw=1.2, c='k')
            ax.text((x1 + x2) * 0.5, base_y + h * 0.8, label,
                    ha='center', va='bottom', color='k', fontsize=14)

        if metric == 'RMSE':
            ax.set_ylabel(f"{metric_labels[m_idx]} $(\\leftarrow)$", fontsize=16)
        else:
            ax.set_ylabel(f"{metric_labels[m_idx]} $(\\rightarrow)$", fontsize=16)
        # ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, fontsize=14, rotation=20)

    plt.tight_layout()
    plt.savefig('./plots/mycoperme_metrics_p.pdf')
    plt.savefig('./plots/mycoperme_metrics_p.eps', dpi=600, bbox_inches='tight')
    plt.close()


plot_r2_rmse_spearman_subplots(baseline_df, baseline_nst_df, baseline_fnst_df)
