import pandas as pd


datasets = ["FreeSolv", "ESOL", "permeability", "Lipo"]
methods = ["nst", "fusion_nst"]

results = {}

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        if dataset in ["FreeSolv", "permeability"]:
            path = f"./results/chemprop_{dataset}_{method}_checkpoint/chemprop_test_performance.csv"
        elif dataset in ["ESOL", "Lipo"]:
            path = f"./results/AttentiveFP_{dataset}_{method}_checkpoint/AttentiveFP_test_performance.csv"
        df = pd.read_csv(path)

        # Baseline and the best RMSE columns
        rmse_cols = [c for c in df.columns if "RMSE" in c]
        baseline = df["Iter0 RMSE"].values
        best_rmse = df[rmse_cols].min(axis=1).values

        # (baseline - best)
        improvement = baseline - best_rmse
        mean, std = improvement.mean(), improvement.std()
        results[dataset][method] = (mean, std)

print("\\begin{table}[!ht]")
print("\\centering")
print("\\caption{Performance improvements (RMSE reduction) of NST and Fusion+NST over their respective baselines.}")
print("\\label{tab:nst_fusion_results}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("Dataset & NST - Baseline & Fusion+NST - Baseline+Fusion \\\\")
print("\\midrule")

for dataset in datasets:
    nst_mean, nst_std = results[dataset]["nst"]
    fusion_mean, fusion_std = results[dataset]["fusion_nst"]
    print(f"{dataset} & ${nst_mean:.3f}\\pm{nst_std:.3f}$ & ${fusion_mean:.3f}\\pm{fusion_std:.3f}$ \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
