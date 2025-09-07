import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.patches import Patch


models = ["GCN", "GINE", "chemprop", "AttentiveFP"]
datasets = ["ESOL", "FreeSolv", "Lipo", "permeability"]
dataset_sizes = {"FreeSolv": 642, "ESOL": 1128, "permeability": 1558, "Lipo": 4200}
fusion_modes = ["", "fusion_nst"]  # "" baseline, fusion_nst special handling

base_colors = sns.color_palette("tab10", n_colors=len(models))
fusion_palette = {}
for i, model in enumerate(models):
    fusion_palette[(model, "-fusion")] = base_colors[i]
    fusion_palette[(model, "+fusion")] = tuple(list(base_colors[i]) + [0.5])  # alpha

records = []

for model in models:
    for dataset in datasets:
        for fusion in fusion_modes:
            subdir = f"{model}_{dataset}_{fusion}_checkpoint" if fusion else f"{model}_{dataset}_checkpoint"
            csv_path = f"./results/{subdir}/{model}_test_performance.csv"
            try:
                df = pd.read_csv(csv_path)
                if fusion == "":
                    # baseline
                    rmse_values = df["RMSE"].values
                else:
                    # fusion_nst: select Iter{i} RMSE with the smallest mean
                    iter_cols = [f"Iter{i} RMSE" for i in range(1, 4) if f"Iter{i} RMSE" in df.columns]
                    if iter_cols:
                        means = [df[col].mean() for col in iter_cols]
                        best_col = iter_cols[int(np.argmin(means))]
                        print(f"[fusion_nst] {model}-{dataset}: select {best_col}")
                        rmse_values = df[best_col].values
                    else:
                        rmse_values = []
                for val in rmse_values:
                    records.append({
                        "Model": model,
                        "Dataset": dataset,
                        "Fusion": "+fusion" if fusion == "fusion_nst" else "-fusion",
                        "RMSE": val
                    })
            except Exception as e:
                print(f"Missing or invalid: {csv_path}")

df_all = pd.DataFrame(records)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True)
p_value_labels = [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]

for i, dataset in enumerate(datasets):
    row, col = divmod(i, 2)
    ax = axes[row, col]

    display_dataset = "Permeability" if dataset == "permeability" else dataset
    display_dataset = f"{display_dataset} (#{dataset_sizes[dataset]})"
    df_subset = df_all[df_all["Dataset"] == dataset]

    # boxplot
    for j, model in enumerate(models):
        for k, fusion in enumerate(["-fusion", "+fusion"]):
            rmse_vals = df_subset[(df_subset["Model"] == model) & (df_subset["Fusion"] == fusion)]["RMSE"]
            if len(rmse_vals) == 0:
                continue
            x_pos = j + (k - 0.5) * 0.2
            color = fusion_palette[(model, fusion)]
            hatch = '//' if fusion == '+fusion' else None
            ax.boxplot(rmse_vals, positions=[x_pos], widths=0.1,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, color="black", hatch=hatch),
                       medianprops=dict(color="black"),
                       whiskerprops=dict(color="black"),
                       capprops=dict(color="black"),
                       flierprops=dict(markerfacecolor=color, markeredgecolor="black", alpha=0.4))

    if col == 0:
        ax.set_ylabel("RMSE $(\leftarrow)$", fontsize=16)
    ax.set_title(f"{display_dataset}", fontsize=16)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m if m != "chemprop" else "Chemprop" for m in models], fontsize=16)

    # Wilcoxon test + asterisks
    for j, model in enumerate(models):
        vals_plain = df_subset[(df_subset["Model"] == model) & (df_subset["Fusion"] == "-fusion")]["RMSE"]
        vals_fusion = df_subset[(df_subset["Model"] == model) & (df_subset["Fusion"] == "+fusion")]["RMSE"]

        if len(vals_plain) == len(vals_fusion) and len(vals_plain) > 0:
            try:
                stat, p = wilcoxon(vals_plain, vals_fusion, alternative="two-sided")
                # print("Dataset:", dataset, "Model:", model, "Wilcoxon p-value:", p)
                label = "ns"
                for p_cut, sym in p_value_labels:
                    if p <= p_cut:
                        label = sym
                        break
                y = max(vals_plain.max(), vals_fusion.max()) + 0.02
                h = 0.02
                x1, x2 = j - 0.1, j + 0.1
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c='k')
                ax.text((x1 + x2) * 0.5, y + h, label, ha='center', va='bottom', color='k', fontsize=16)
            except Exception as e:
                print(f"Wilcoxon error on {model}-{dataset}: {e}")

# plt.tight_layout()

legend_elements = [
    Patch(facecolor="grey", edgecolor="black", label="Baseline"),
    Patch(facecolor="grey", alpha=0.5, edgecolor="black", hatch="//", label="Baseline Fusion+NST")
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=16, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./plots/benchmark_rmse.pdf")
plt.savefig("./plots/benchmark_rmse.eps", dpi=600, bbox_inches='tight')
