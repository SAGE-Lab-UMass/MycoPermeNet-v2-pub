"QSAR modeling for permeability dataset."

import os
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_tools.pyg_chemprop_utils import scaffold_balanced_split

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def build_features(smiles_list, smiles_to_rdkit, smiles_to_ecfp):
    """
    Build feature matrix by concatenating RDKit descriptors and ECFP bits.
      X = [rdkit_features, ecfp_bits]
    """
    rdkit_features = []
    ecfp_features = []

    for smi in smiles_list:
        rdkit_features.append(smiles_to_rdkit[smi])
        ecfp_features.append(smiles_to_ecfp[smi])

    rdkit_features = np.stack(rdkit_features, axis=0)
    ecfp_features = np.stack(ecfp_features, axis=0)

    X = np.concatenate([rdkit_features, ecfp_features], axis=1)
    return X


class SimpleDataset:
    def __init__(self, smiles, y):
        self.smiles = smiles
        self.y = y


parser = argparse.ArgumentParser(description="QSAR modeling for permeability dataset")
parser.add_argument(
    "--model",
    type=str,
    default="XGBoost",
    choices=["XGBoost", "RandomForest"],
    help="QSAR model",
)
args = parser.parse_args()

random.seed(42)  # For reproducibility
mlp_seeds = random.sample(range(100, 200), 49)

results = []
val_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = f'./results/{args.model}_permeability_checkpoint'
os.makedirs(save_dir, exist_ok=True)

train_df = pd.read_csv('./data/train_scaffold_split.csv')
test_df = pd.read_csv('./data/test_scaffold_split.csv')

# rdkit = pd.read_excel("./data/fingerprints_and_descriptors.xlsx", sheet_name="NormDescriptorsSmile")
rdkit = pd.read_csv("./data/preprocessed_labeled_descriptors.csv")
ecfp = pd.read_excel("./data/fingerprints_and_descriptors.xlsx", sheet_name="ECFP")

smiles_to_rdkit = {row['Smiles']: row.iloc[1:].astype(float).values for _, row in rdkit.iterrows()}
smiles_to_ecfp = {row['Smiles']: row.iloc[1:].astype(float).values for _, row in ecfp.iterrows()}

train_dataset = [
    SimpleDataset(row["Smiles"], float(row["MTB Standardized Residuals"])) for _, row in train_df.iterrows()
]

test_smiles = test_df["Smiles"].tolist()
X_test = build_features(test_smiles, smiles_to_rdkit, smiles_to_ecfp)
y_test = test_df["MTB Standardized Residuals"].values.astype(float)

for i, mlp_seed in enumerate(mlp_seeds):
    print(f"Running experiment with MLP seed {mlp_seed}")
    set_seed(mlp_seed)

    train_idx, val_idx, filtered_dataset = scaffold_balanced_split(
        train_dataset, val_ratio=0.2, seed=mlp_seed
    )

    train_smiles = [filtered_dataset[j].smiles for j in train_idx]
    val_smiles = [filtered_dataset[j].smiles for j in val_idx]

    y_train = np.array([filtered_dataset[j].y for j in train_idx], dtype=float)
    y_val = np.array([filtered_dataset[j].y for j in val_idx], dtype=float)

    X_train = build_features(train_smiles, smiles_to_rdkit, smiles_to_ecfp)
    X_val = build_features(val_smiles, smiles_to_rdkit, smiles_to_ecfp)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    if args.model == "XGBoost":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            objective="reg:squarederror",
            random_state=mlp_seed,
            n_jobs=-1,
            tree_method="hist",
        )
    elif args.model == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=mlp_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model {args.model}")

    if i == 0:
        print("X_train shape:", X_train.shape)

    model.fit(X_train, y_train_scaled)

    y_val_pred_scaled = model.predict(X_val)
    y_test_pred_scaled = model.predict(X_test)

    y_val_pred = y_scaler.inverse_transform(
        y_val_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_test_pred = y_scaler.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()

    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    val_spearman = spearmanr(y_val, y_val_pred).correlation

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_spearman = spearmanr(y_test, y_test_pred).correlation

    print(
        f"Val: R2: {val_r2:.4f} | RMSE (orig): {val_rmse:.4f} | Spearman: {val_spearman:.4f}"
    )
    print(
        f"Test: R2: {test_r2:.4f} | RMSE (orig): {test_rmse:.4f} | Spearman: {test_spearman:.4f}"
    )

    val_results.append({
        "Seed": mlp_seed,
        "R2": val_r2,
        "RMSE": val_rmse,
        "Spearman": val_spearman,
    })

    results.append({
        "Seed": mlp_seed,
        "R2": test_r2,
        "RMSE": test_rmse,
        "Spearman": test_spearman,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, f'{args.model}_test_performance.csv'), index=False)

val_results_df = pd.DataFrame(val_results)
val_results_df.to_csv(os.path.join(save_dir, f'{args.model}_val_performance.csv'), index=False)
