import pandas as pd
import matplotlib.pyplot as plt

# Load data
dataset = 'permeability'
df = pd.read_csv(f'./results/chemprop_{dataset}_mine_weight_checkpoint/chemprop_mine_weight_performance_new.csv')

# Group statistics
stats = df.groupby('Combine Weight').agg({
    'Validation R2': ['mean', 'std'],
    'Validation RMSE': ['mean', 'std'],
    'Validation Spearman': ['mean', 'std'],
}).reset_index()
stats.columns = ['Combine Weight', 'Mean Val R2', 'Std Val R2',
                 'Mean Val RMSE', 'Std Val RMSE',
                 'Mean Val Spearman', 'Std Val Spearman']

# Load baseline
baseline_df = pd.read_csv(f'./results/chemprop_{dataset}_baseline_weight_checkpoint/chemprop_baseline_weight_performance.csv')
baseline_mean_val_r2 = baseline_df['Validation R2'].mean()
baseline_std_val_r2 = baseline_df['Validation R2'].std()
baseline_mean_val_rmse = baseline_df['Validation RMSE'].mean()
baseline_std_val_rmse = baseline_df['Validation RMSE'].std()
baseline_mean_val_spearman = baseline_df['Validation Spearman'].mean()
baseline_std_val_spearman = baseline_df['Validation Spearman'].std()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# === R2 Plot ===
axes[0].errorbar(stats['Combine Weight'], stats['Mean Val R2'],
                 yerr=stats['Std Val R2'],
                 fmt='o', color='blue', capsize=3, alpha=0.8, label='MINE-based')

axes[0].errorbar(0, baseline_mean_val_r2,
                 yerr=baseline_std_val_r2,
                 fmt='o', color='black', capsize=3, alpha=0.8, label='Baseline')

axes[0].axhline(y=baseline_mean_val_r2, color='black', linestyle='--', linewidth=1)

axes[0].set_xlabel('Combined Weight', fontsize=16)
axes[0].set_ylabel('Validation $R^2 \\; (\\rightarrow)$', fontsize=16)
# axes[0].set_title('Validation $R^2$ vs Combined Weight', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=14)
# axes[0].set_ylim(0.4, 0.7)
# axes[0].set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
axes[0].text(0, 1.08, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# === RMSE Plot ===
axes[1].errorbar(stats['Combine Weight'], stats['Mean Val RMSE'],
                 yerr=stats['Std Val RMSE'],
                 fmt='o', color='red', capsize=3, alpha=0.8, label='MINE-based')

axes[1].errorbar(0, baseline_mean_val_rmse,
                 yerr=baseline_std_val_rmse,
                 fmt='o', color='black', capsize=3, alpha=0.8, label='Baseline')

axes[1].axhline(y=baseline_mean_val_rmse, color='black', linestyle='--', linewidth=1)

axes[1].set_xlabel('Combined Weight', fontsize=16)
axes[1].set_ylabel('Validation RMSE $(\\leftarrow)$', fontsize=16)
# axes[1].set_title('Validation RMSE vs Combined Weight', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=14, loc='upper left')
axes[1].text(0, 1.08, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# === Spearman Plot ===
axes[2].errorbar(stats['Combine Weight'], stats['Mean Val Spearman'],
                 yerr=stats['Std Val Spearman'],
                 fmt='o', color='green', capsize=3, alpha=0.8, label='MINE-based')

axes[2].errorbar(0, baseline_mean_val_spearman,
                 yerr=baseline_std_val_spearman,
                 fmt='o', color='black', capsize=3, alpha=0.8, label='Baseline')

axes[2].axhline(y=baseline_mean_val_spearman, color='black', linestyle='--', linewidth=1)

axes[2].set_xlabel('Combined Weight', fontsize=16)
axes[2].set_ylabel('Validation Spearman $(\\rightarrow)$', fontsize=16)
# axes[2].set_title('Validation Spearman vs Combined Weight', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=14)
axes[2].text(0, 1.08, 'C', transform=axes[2].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig(f'./plots/{dataset}_mine_weight_search.pdf')
plt.savefig(f'./plots/{dataset}_mine_weight_search.eps', dpi=600, bbox_inches='tight')

# Console stats
print("\nStatistical Summary:")
print(stats.to_string(index=False))

print("\nSorted by Mean Val R2 (Descending):")
print(stats.sort_values('Mean Val R2', ascending=False).to_string(index=False))

# 0.592415 (FreeSolv) 0.785176 0.598658
# 0.139494 (ESOL) 0.065052
# 0.046450 (Lipo) 0.020584
# 0.058084 (permeability)
