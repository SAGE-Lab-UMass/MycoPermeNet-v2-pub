import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# Compute dynamic xlim
def compute_limits(series, margin=0.05):
    min_val = series.min()
    max_val = series.max()
    xlim = (min_val - margin, max_val + margin)
    return xlim


# Load baseline data
allVar_df = pd.read_csv('./results/chemprop_all_var_checkpoint/chemprop_all_var_performance.csv')
dataVar_df = pd.read_csv('./results/chemprop_data_var_checkpoint/chemprop_data_var_performance.csv')
torchVar_df = pd.read_csv('./results/chemprop_torch_var_checkpoint/chemprop_torch_var_performance.csv')
crossVar_df = pd.read_csv('./results/chemprop_cross_var_checkpoint/chemprop_cross_var_performance.csv')

# Combine all DataFrames
df_all = pd.concat([allVar_df, dataVar_df, torchVar_df, crossVar_df], ignore_index=True)

# Compute dynamic limits for each metric
r2_xlim = compute_limits(df_all['Test R2'])
rmse_xlim = compute_limits(df_all['Test RMSE'])
spearmans_xlim = compute_limits(df_all['Test Spearman'])
r2_ylim = (0, 20)
rmse_ylim = (0, 20)
spearmans_ylim = (0, 45)

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(18, 22))

# === allVar ===
sns.kdeplot(allVar_df['Test R2'], ax=axes[0, 0], label='Density', color='blue', fill=True, alpha=0.3)
sns.rugplot(allVar_df['Test R2'], ax=axes[0, 0], label='Samples', color='blue', height=0.05, alpha=0.8)

axes[0, 0].axvline(allVar_df['Test R2'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')

axes[0, 0].set_xlim(r2_xlim)
axes[0, 0].set_ylim(r2_ylim)
# axes[0, 0].set_xlabel('Test $R^2$', fontsize=16)
axes[0, 0].set_ylabel('Density', fontsize=16)
axes[0, 0].grid(True, alpha=0.3)
# axes[0, 0].legend(fontsize=14)
axes[0, 0].text(0, 1.08, 'A', transform=axes[0, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(allVar_df['Test RMSE'], ax=axes[0, 1], label='Density', color='red', fill=True, alpha=0.3)
sns.rugplot(allVar_df['Test RMSE'], ax=axes[0, 1], label='Samples', color='red', height=0.05, alpha=0.8)

axes[0, 1].axvline(allVar_df['Test RMSE'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

axes[0, 1].set_xlim(rmse_xlim)
axes[0, 1].set_ylim(rmse_ylim)
# axes[0, 1].set_xlabel('Test RMSE', fontsize=16)
# axes[0, 1].set_ylabel('Density', fontsize=16)
axes[0, 1].grid(True, alpha=0.3)
# axes[0, 1].legend(fontsize=14)
# axes[0, 1].text(0, 1.08, 'B', transform=axes[0, 1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(allVar_df['Test Spearman'], ax=axes[0, 2], label='Density', color='green', fill=True, alpha=0.3)
sns.rugplot(allVar_df['Test Spearman'], ax=axes[0, 2], label='Samples', color='green', height=0.05, alpha=0.8)

axes[0, 2].axvline(allVar_df['Test Spearman'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')

axes[0, 2].set_xlim(spearmans_xlim)
axes[0, 2].set_ylim(spearmans_ylim)
# axes[0, 2].set_xlabel('Test Spearman', fontsize=16)
# axes[0, 2].set_ylabel('Density', fontsize=16)
axes[0, 2].grid(True, alpha=0.3)
# axes[0, 2].legend(fontsize=14)

# === dataVar ===
sns.kdeplot(dataVar_df['Test R2'], ax=axes[1, 0], label='Density', color='blue', fill=True, alpha=0.3)
sns.rugplot(dataVar_df['Test R2'], ax=axes[1, 0], label='Samples', color='blue', height=0.05, alpha=0.8)

axes[1, 0].axvline(dataVar_df['Test R2'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')

axes[1, 0].set_xlim(r2_xlim)
axes[1, 0].set_ylim(r2_ylim)
# axes[1, 0].set_xlabel('Test $R^2$', fontsize=16)
axes[1, 0].set_ylabel('Density', fontsize=16)
axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].legend(fontsize=14)
axes[1, 0].text(0, 1.08, 'B', transform=axes[1, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(dataVar_df['Test RMSE'], ax=axes[1, 1], label='Density', color='red', fill=True, alpha=0.3)
sns.rugplot(dataVar_df['Test RMSE'], ax=axes[1, 1], label='Samples', color='red', height=0.05, alpha=0.8)

axes[1, 1].axvline(dataVar_df['Test RMSE'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

axes[1, 1].set_xlim(rmse_xlim)
axes[1, 1].set_ylim(rmse_ylim)
# axes[1, 1].set_xlabel('Test RMSE', fontsize=16)
# axes[1, 1].set_ylabel('Density', fontsize=16)
axes[1, 1].grid(True, alpha=0.3)
# axes[1, 1].legend(fontsize=14)
# axes[1, 1].text(0, 1.08, 'D', transform=axes[1, 1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(dataVar_df['Test Spearman'], ax=axes[1, 2], label='Density', color='green', fill=True, alpha=0.3)
sns.rugplot(dataVar_df['Test Spearman'], ax=axes[1, 2], label='Samples', color='green', height=0.05, alpha=0.8)

axes[1, 2].axvline(dataVar_df['Test Spearman'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')

axes[1, 2].set_xlim(spearmans_xlim)
axes[1, 2].set_ylim(spearmans_ylim)
# axes[1, 2].set_xlabel('Test Spearman', fontsize=16)
# axes[1, 2].set_ylabel('Density', fontsize=16)
axes[1, 2].grid(True, alpha=0.3)
# axes[1, 2].legend(fontsize=14)

# === torchVar ===
sns.kdeplot(torchVar_df['Test R2'], ax=axes[2, 0], label='Density', color='blue', fill=True, alpha=0.3)
sns.rugplot(torchVar_df['Test R2'], ax=axes[2, 0], label='Samples', color='blue', height=0.05, alpha=0.8)

axes[2, 0].axvline(torchVar_df['Test R2'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')

axes[2, 0].set_xlim(r2_xlim)
axes[2, 0].set_ylim(r2_ylim)
# axes[2, 0].set_xlabel('$R^2$ $(\\rightarrow)$', fontsize=16)
axes[2, 0].set_ylabel('Density', fontsize=16)
axes[2, 0].grid(True, alpha=0.3)
# axes[2, 0].legend(fontsize=14)
axes[2, 0].text(0, 1.08, 'C', transform=axes[2, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(torchVar_df['Test RMSE'], ax=axes[2, 1], label='Density', color='red', fill=True, alpha=0.3)
sns.rugplot(torchVar_df['Test RMSE'], ax=axes[2, 1], label='Samples', color='red', height=0.05, alpha=0.8)

axes[2, 1].axvline(torchVar_df['Test RMSE'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

axes[2, 1].set_xlim(rmse_xlim)
axes[2, 1].set_ylim(rmse_ylim)
# axes[2, 1].set_xlabel('RMSE $(\\leftarrow)$', fontsize=16)
# axes[2, 1].set_ylabel('Density', fontsize=16)
axes[2, 1].grid(True, alpha=0.3)
# axes[2, 1].legend(fontsize=14)
# axes[2, 1].text(0, 1.08, 'F', transform=axes[2, 1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(torchVar_df['Test Spearman'], ax=axes[2, 2], label='Density', color='green', fill=True, alpha=0.3)
sns.rugplot(torchVar_df['Test Spearman'], ax=axes[2, 2], label='Samples', color='green', height=0.05, alpha=0.8)

axes[2, 2].axvline(torchVar_df['Test Spearman'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')

axes[2, 2].set_xlim(spearmans_xlim)
axes[2, 2].set_ylim(spearmans_ylim)
# axes[2, 2].set_xlabel('Spearman $(\\rightarrow)$', fontsize=16)
# axes[2, 2].set_ylabel('Density', fontsize=16)
axes[2, 2].grid(True, alpha=0.3)
# axes[2, 2].legend(fontsize=14)

# === crossVar ===
sns.kdeplot(crossVar_df['Test R2'], ax=axes[3, 0], label='Density', color='blue', fill=True, alpha=0.3)
sns.rugplot(crossVar_df['Test R2'], ax=axes[3, 0], label='Samples', color='blue', height=0.05, alpha=0.8)

axes[3, 0].axvline(crossVar_df['Test R2'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')

axes[3, 0].set_xlim(r2_xlim)
axes[3, 0].set_ylim(r2_ylim)
axes[3, 0].set_xlabel('$R^2$ $(\\rightarrow)$', fontsize=16)
axes[3, 0].set_ylabel('Density', fontsize=16)
axes[3, 0].grid(True, alpha=0.3)
# axes[3, 0].legend(fontsize=14)
axes[3, 0].text(0, 1.08, 'D', transform=axes[3, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(crossVar_df['Test RMSE'], ax=axes[3, 1], label='Density', color='red', fill=True, alpha=0.3)
sns.rugplot(crossVar_df['Test RMSE'], ax=axes[3, 1], label='Samples', color='red', height=0.05, alpha=0.8)

axes[3, 1].axvline(crossVar_df['Test RMSE'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

axes[3, 1].set_xlim(rmse_xlim)
axes[3, 1].set_ylim(rmse_ylim)
axes[3, 1].set_xlabel('RMSE $(\\leftarrow)$', fontsize=16)
# axes[3, 1].set_ylabel('Density', fontsize=16)
axes[3, 1].grid(True, alpha=0.3)
# axes[3, 1].legend(fontsize=14)
# axes[3, 1].text(0, 1.08, 'D', transform=axes[1, 1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

sns.kdeplot(crossVar_df['Test Spearman'], ax=axes[3, 2], label='Density', color='green', fill=True, alpha=0.3)
sns.rugplot(crossVar_df['Test Spearman'], ax=axes[3, 2], label='Samples', color='green', height=0.05, alpha=0.8)

axes[3, 2].axvline(crossVar_df['Test Spearman'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')

axes[3, 2].set_xlim(spearmans_xlim)
axes[3, 2].set_ylim(spearmans_ylim)
axes[3, 2].set_xlabel('Spearman $(\\rightarrow)$', fontsize=16)
# axes[3, 2].set_ylabel('Density', fontsize=16)
axes[3, 2].grid(True, alpha=0.3)
# axes[3, 2].legend(fontsize=14)


# Only keep axis labels on leftmost and bottom plots
for i in range(4):
    for j in range(3):
        ax = axes[i, j]
        if j != 0:
            ax.set_ylabel('')
        if i != 3:
            ax.set_xlabel('')

# Manual legend construction
density_patch = mpatches.Patch(facecolor='grey', alpha=0.5, edgecolor="black", label='Density')
sample_rug = mlines.Line2D([], [], color='black', label='Samples', linestyle='', marker='|', markersize=10)
mean_line = mlines.Line2D([], [], color='black', label='Mean', linestyle='--', linewidth=2)

# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 1, 0.985])
fig.legend(handles=[density_patch, sample_rug, mean_line],
           loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=3, fontsize=16, frameon=True)

plt.savefig('./plots/chemprop_randomness.pdf')
plt.savefig('./plots/chemprop_randomness.eps', dpi=600, bbox_inches='tight')
