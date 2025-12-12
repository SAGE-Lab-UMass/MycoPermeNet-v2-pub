import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

base_dir = "./results"

percents = [0.25, 0.5, 0.75, 1.0]
x_vals = [p * 100 for p in percents]  # 25, 50, 75, 100

base_colors = sns.color_palette("tab10", n_colors=5)


def load_curve_permeability():
    baseline_means, baseline_stds = [], []
    fusion_means, fusion_stds = [], []
    delta_means, delta_stds = [], []

    for p in percents:
        if p < 1.0:
            base_folder = f"chemprop_permeability_{p}_checkpoint"
            fusion_folder = f"chemprop_permeability_fusion_nst_{p}_checkpoint"
        else:
            base_folder = "chemprop_permeability_checkpoint"
            fusion_folder = "chemprop_permeability_fusion_nst_checkpoint"

        base_path = os.path.join(base_dir, base_folder, "chemprop_test_performance.csv")
        fusion_path = os.path.join(base_dir, fusion_folder, "chemprop_test_performance.csv")

        # baseline
        df_base = pd.read_csv(base_path)
        base_rmse = df_base["RMSE"].values

        # fusion + NST
        df_fusion = pd.read_csv(fusion_path)
        if p in (0.25, 0.5):
            col = "Iter3 RMSE"
        else:   # 0.75, 1.0
            col = "Iter2 RMSE"
        fusion_rmse = df_fusion[col].values

        assert len(base_rmse) == len(fusion_rmse)

        baseline_means.append(base_rmse.mean())
        baseline_stds.append(base_rmse.std(ddof=1))
        fusion_means.append(fusion_rmse.mean())
        fusion_stds.append(fusion_rmse.std(ddof=1))

        delta = base_rmse - fusion_rmse
        delta_means.append(delta.mean())
        delta_stds.append(delta.std(ddof=1))

    return (np.array(baseline_means),
            np.array(baseline_stds),
            np.array(fusion_means),
            np.array(fusion_stds),
            np.array(delta_means),
            np.array(delta_stds))


def load_curve_lipo():
    baseline_means, baseline_stds = [], []
    fusion_means, fusion_stds = [], []
    delta_means, delta_stds = [], []

    for p in percents:
        if p < 1.0:
            base_folder = f"AttentiveFP_Lipo_{p}_checkpoint"
            fusion_folder = f"AttentiveFP_Lipo_fusion_nst_{p}_checkpoint"
        else:
            base_folder = "AttentiveFP_Lipo_checkpoint"
            fusion_folder = "AttentiveFP_Lipo_fusion_nst_checkpoint"

        base_path = os.path.join(base_dir, base_folder, "AttentiveFP_test_performance.csv")
        fusion_path = os.path.join(base_dir, fusion_folder, "AttentiveFP_test_performance.csv")

        df_base = pd.read_csv(base_path)
        base_rmse = df_base["RMSE"].values

        df_fusion = pd.read_csv(fusion_path)
        if p == 0.25 or p == 1.0:
            col = "Iter3 RMSE"
        elif p == 0.5:
            col = "Iter2 RMSE"
        elif p == 0.75:
            col = "Iter1 RMSE"
        else:
            raise ValueError(f"Unexpected percent {p}")
        fusion_rmse = df_fusion[col].values

        assert len(base_rmse) == len(fusion_rmse)

        baseline_means.append(base_rmse.mean())
        baseline_stds.append(base_rmse.std(ddof=1))
        fusion_means.append(fusion_rmse.mean())
        fusion_stds.append(fusion_rmse.std(ddof=1))

        delta = base_rmse - fusion_rmse
        delta_means.append(delta.mean())
        delta_stds.append(delta.std(ddof=1))

    return (np.array(baseline_means),
            np.array(baseline_stds),
            np.array(fusion_means),
            np.array(fusion_stds),
            np.array(delta_means),
            np.array(delta_stds))


(perm_base_mean, perm_base_std,
 perm_fus_mean, perm_fus_std,
 perm_delta_mean, perm_delta_std) = load_curve_permeability()

(lipo_base_mean, lipo_base_std,
 lipo_fus_mean, lipo_fus_std,
 lipo_delta_mean, lipo_delta_std) = load_curve_lipo()

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 6),
    sharex=True,
    # sharey="row",
    height_ratios=[1.5, 1],
)

# RMSE curves
ax_perm = axes[0, 0]
ax_lipo = axes[0, 1]

# Permeability RMSE
ax_perm.errorbar(
    x_vals,
    perm_base_mean,
    yerr=perm_base_std,
    marker="o",
    linestyle="-",
    label="Baseline",
    color=base_colors[0],
    capsize=3,
    alpha=0.8,
)
ax_perm.errorbar(
    x_vals,
    perm_fus_mean,
    yerr=perm_fus_std,
    marker="s",
    linestyle="-",
    label="Baseline Fusion+NST",
    color=base_colors[3],
    capsize=3,
    alpha=0.8,
)
ax_perm.set_ylabel("RMSE $(\\leftarrow)$", fontsize=13)
ax_perm.set_title("Permeability", fontsize=14)
ax_perm.grid(True, linestyle=":", alpha=0.5)
ax_perm.legend(fontsize=10)

# Lipo RMSE
ax_lipo.errorbar(
    x_vals,
    lipo_base_mean,
    yerr=lipo_base_std,
    marker="o",
    linestyle="-",
    label="Baseline",
    color=base_colors[0],
    capsize=3,
    alpha=0.8,
)
ax_lipo.errorbar(
    x_vals,
    lipo_fus_mean,
    yerr=lipo_fus_std,
    marker="s",
    linestyle="-",
    label="Baseline Fusion+NST",
    color=base_colors[3],
    capsize=3,
    alpha=0.8,
)
ax_lipo.set_title("Lipo", fontsize=14)
ax_lipo.grid(True, linestyle=":", alpha=0.5)
ax_lipo.legend(fontsize=10)

# Mean RMSE reduction bar chart
ax_perm_bar = axes[1, 0]
ax_lipo_bar = axes[1, 1]

# Permeability reduction
ax_perm_bar.bar(
    x_vals,
    perm_delta_mean,
    # yerr=perm_delta_std,
    color=base_colors[4],
    alpha=0.8,
    width=6,
    # capsize=3,
)
ax_perm_bar.set_ylabel(r"$\overline{\Delta \mathrm{RMSE}}$", fontsize=13)
ax_perm_bar.grid(True, linestyle=":", alpha=0.5)

# Lipo reduction
ax_lipo_bar.bar(
    x_vals,
    lipo_delta_mean,
    # yerr=lipo_delta_std,
    color=base_colors[4],
    alpha=0.8,
    width=6,
    # capsize=3,
)
ax_lipo_bar.grid(True, linestyle=":", alpha=0.5)

for ax in axes[1, :]:
    ax.set_xlabel("Training data used (%)", fontsize=13)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{int(x)}%" for x in x_vals])

plt.tight_layout()
plt.savefig("./plots/rmse_percent_gain_bar.pdf")
