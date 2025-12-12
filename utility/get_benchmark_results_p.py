import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


models = ["GCN", "GINE", "chemprop", "AttentiveFP"]
datasets = ["ESOL", "FreeSolv", "Lipo", "permeability"]
fusion_modes = ["", "fusion_nst"]  # "" is baseline, fusion_nst is special handling

# {model: {fusion: {dataset: list of RMSE}}}
results = {m: {f: {} for f in fusion_modes} for m in models}

for model in models:
    for dataset in datasets:
        for fusion in fusion_modes:
            if dataset == "permeability":
                subdir = f"{model}_{dataset}_{fusion}_checkpoint" if fusion else f"{model}_{dataset}_checkpoint"
            else:
                subdir = f"{model}_{dataset}_{fusion}_checkpoint" if fusion else f"{model}_{dataset}_nst_checkpoint"
            csv_path = f"./results/{subdir}/{model}_test_performance.csv"
            try:
                df = pd.read_csv(csv_path)
                if fusion == "":
                    # baseline: directly use RMSE column
                    if dataset == "permeability":
                        results[model][fusion][dataset] = df["RMSE"].values
                    else:
                        results[model][fusion][dataset] = df["Iter0 RMSE"].values
                else:
                    # fusion_nst: pick the Iter{i} RMSE column with smallest mean
                    iter_cols = [f"Iter{i} RMSE" for i in range(1, 4) if f"Iter{i} RMSE" in df.columns]
                    if iter_cols:
                        means = [df[col].mean() for col in iter_cols]
                        best_col = iter_cols[int(np.argmin(means))]
                        results[model][fusion][dataset] = df[best_col].values
                    else:
                        results[model][fusion][dataset] = None
            except Exception:
                results[model][fusion][dataset] = None

dataset_names = ["ESOL", "FreeSolv", "Lipo", "Permeability"]

# Get the best RMSE for each dataset across all models
best_per_dataset = {d: float('inf') for d in datasets}
for dataset in datasets:
    for model in models:
        for fusion in fusion_modes:
            values = results[model][fusion].get(dataset)
            if values is not None:
                mean_val = np.mean(values)
                if mean_val < best_per_dataset[dataset]:
                    best_per_dataset[dataset] = mean_val

print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Test RMSE ($\\downarrow$) of different models with and without fusion. Mean ± std over repeated runs.}")
print("\\label{tab:rmse_fusion}")
print("\\begin{tabular}{ll" + "c" * len(datasets) + "}")
print("\\toprule")
print(" & Model & " + " & ".join(dataset_names) + " \\\\")
print("\\midrule")

for model in models:
    for fusion in fusion_modes:
        label = "+ Fusion" if fusion else ""
        model_name = "Chemprop" if model == "chemprop" else model
        row = [f"{model_name} {label}"]
        for dataset in datasets:
            values = results[model][fusion].get(dataset)
            if values is None:
                row.append("--")
            else:
                values = np.array(values)
                mean_val = values.mean()
                cell = f"{mean_val:.3f} $\\pm$ {values.std(ddof=1):.3f}"
                # Bold the best
                if np.isclose(mean_val, best_per_dataset[dataset]):
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
        print(" & " + " & ".join(row) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")


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


stats = {m: {d: {"effect": None, "p_raw": None, "p_adj": None}
             for d in datasets}
         for m in models}

p_list = []
pairs = []

for model in models:
    for dataset in datasets:
        base_vals = results[model][""].get(dataset)
        fusion_vals = results[model]["fusion_nst"].get(dataset)
        if base_vals is None or fusion_vals is None:
            continue

        base_vals = np.asarray(base_vals)
        fusion_vals = np.asarray(fusion_vals)

        # Wilcoxon signed-rank test (two-sided)
        stat, p_raw = wilcoxon(base_vals, fusion_vals, alternative="two-sided")
        effect = rank_biserial_effect(base_vals, fusion_vals)

        stats[model][dataset]["effect"] = effect
        stats[model][dataset]["p_raw"] = p_raw

        p_list.append(p_raw)
        pairs.append((model, dataset))


reject, p_adj_all, _, _ = multipletests(p_list, method="holm")

for (model, dataset), p_adj in zip(pairs, p_adj_all):
    stats[model][dataset]["p_adj"] = p_adj


lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"    \centering")
lines.append(r"    \caption{Wilcoxon signed-rank tests on RMSE between baseline and Fusion+NST across GNN encoders. Effect sizes are rank-biserial correlations.}")
lines.append(r"    \label{tab:s15_effect_pvalues}")
lines.append(r"    \begin{tabular}{llcccc}")
lines.append(r"        \toprule")
lines.append(r"        GNN encoder &  & " + " & ".join(dataset_names) + r" \\")
lines.append(r"        \midrule")

name_map = {"chemprop": "Chemprop", "AttentiveFP": "Attentive FP",
            "GCN": "GCN", "GINE": "GINE"}

for model in models:
    model_name = name_map.get(model, model)

    row_eff = [r"\multirow{3}{*}{" + model_name + r"}",
               r"Effect size $|\Delta|$"]
    row_raw = ["", r"Raw $p$"]
    row_adj = ["", r"Adjusted $p$"]

    for dataset in datasets:
        s = stats[model][dataset]
        if s["effect"] is None:
            cell_eff = cell_raw = cell_adj = r"--"
        else:
            cell_eff = f"{s['effect']:.3f}"
            cell_raw = f"{s['p_raw']:.2e}"
            cell_adj = f"{s['p_adj']:.2e}"

        row_eff.append(cell_eff)
        row_raw.append(cell_raw)
        row_adj.append(cell_adj)

    lines.append("        " + " & ".join(row_eff) + r" \\")
    lines.append("        " + " & ".join(row_raw) + r" \\")
    lines.append("        " + " & ".join(row_adj) + r" \\")
    lines.append(r"        \midrule")

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"\end{table}")

latex_s15 = "\n".join(lines)
print(latex_s15)
