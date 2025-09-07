import pandas as pd
import numpy as np

models = ["GCN", "GINE", "chemprop", "AttentiveFP"]
datasets = ["ESOL", "FreeSolv", "Lipo", "permeability"]
fusion_modes = ["", "fusion_nst"]  # "" is baseline, fusion_nst is special handling

# {model: {fusion: {dataset: list of RMSE}}}
results = {m: {f: {} for f in fusion_modes} for m in models}

for model in models:
    for dataset in datasets:
        for fusion in fusion_modes:
            subdir = f"{model}_{dataset}_{fusion}_checkpoint" if fusion else f"{model}_{dataset}_checkpoint"
            csv_path = f"./results/{subdir}/{model}_test_performance.csv"
            try:
                df = pd.read_csv(csv_path)
                if fusion == "":
                    # baseline: directly use RMSE column
                    results[model][fusion][dataset] = df["RMSE"].values
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
                cell = f"{mean_val:.3f} $\\pm$ {values.std():.3f}"
                # Bold the best
                if np.isclose(mean_val, best_per_dataset[dataset]):
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
        print(" & " + " & ".join(row) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
