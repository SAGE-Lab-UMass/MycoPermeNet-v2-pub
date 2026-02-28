"The script for evaluating an unlabeled dataset using the MycoPermeNet-v1 and v2"
import torch
import copy
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from data_tools.evaluate_utils import (get_eval_dataset,
                                       get_eval_dataloaders,
                                       get_eval_representations,
                                       compute_descriptors_for_smiles)

from models.chemprop import DMPNNEncoder
from models.mlp import MLPRegressor


test_df = pd.read_csv("./data/test_scaffold_split.csv")

# # Use your dataset below
# test_df = pd.read_excel("./data/test_compounds.xlsx")
# test_df.rename(columns={"SMILES": "Smiles"}, inplace=True)

feature_scaler_v1 = torch.load("./best_MPN/mlp_feature_scaler_v1.pt", map_location="cpu")
feature_scaler_v2 = torch.load("./best_MPN/mlp_feature_scaler_v2.pt", map_location="cpu")

test_dataset = get_eval_dataset(test_df)
test_loader_v1 = get_eval_dataloaders(copy.deepcopy(test_dataset),
                                      batch_size=64,
                                      feature_scaler=feature_scaler_v1,
                                      target_scaler=None)
test_loader_v2 = get_eval_dataloaders(copy.deepcopy(test_dataset),
                                      batch_size=64,
                                      feature_scaler=feature_scaler_v2,
                                      target_scaler=None)

device = "cuda" if torch.cuda.is_available() else "cpu"
best_gnn_v1 = DMPNNEncoder(hidden_size=300, node_fdim=133, edge_fdim=14,
                           depth=3, dropout=0).to(device)
best_gnn_v1.load_state_dict(torch.load("./best_MPN/best_GNN_v1.pt", map_location=device))

best_gnn_v2 = DMPNNEncoder(hidden_size=300, node_fdim=133, edge_fdim=14,
                           depth=3, dropout=0).to(device)
best_gnn_v2.load_state_dict(torch.load("./best_MPN/best_GNN_v2.pt", map_location=device))

X_test_v1 = get_eval_representations(best_gnn_v1, test_loader_v1, device)
X_test_v2 = get_eval_representations(best_gnn_v2, test_loader_v2, device, smile=True)

# X_test_v2.to_csv("./best_MPN/emb/emb_200_compounds_v2.csv", index=False)
# print(X_test_v2.shape)

# Fuse RDKit descriptors to the MPN-v2 embeddings
with open("./best_MPN/descriptors_minmax_scaler.pkl", "rb") as f:
    pack = pickle.load(f)

desc_cols = pack["columns"]
desc_scaler = pack["scaler"]

smiles_list = X_test_v2["Smiles"].tolist()
df_raw_desc = compute_descriptors_for_smiles(smiles_list, desc_cols)

# raw matrix
desc_raw = df_raw_desc[desc_cols].to_numpy(dtype=np.float64)

# Use the same data_min from training to handle NaN/inf in test descriptors,
# ensuring consistent scaling
data_min = desc_scaler.data_min_.astype(np.float64)  # shape: (n_features,)
nan_mask = ~np.isfinite(desc_raw)
if nan_mask.any():
    desc_raw[nan_mask] = np.take(data_min, np.where(nan_mask)[1])

desc_norm = desc_scaler.transform(desc_raw)  # shape: (n_samples, n_desc)
df_desc_norm = pd.DataFrame(desc_norm, columns=desc_cols)

X_test_v2 = pd.concat(
    [X_test_v2.reset_index(drop=True),
     df_desc_norm.reset_index(drop=True)],
    axis=1
)

target_scaler_v1 = torch.load('./best_MPN/mlp_target_scaler_v1.pt', map_location='cpu')
target_scaler_v2 = torch.load('./best_MPN/mlp_target_scaler_v2.pt', map_location='cpu')
feature_cols_v1 = joblib.load('./best_MPN/mlp_feature_cols_v1.pkl')
feature_cols_v2 = joblib.load('./best_MPN/mlp_feature_cols_v2.pkl')

mlp_v1 = MLPRegressor(input_dim=len(feature_cols_v1), hidden_layer_sizes=(128, 64, 16)).to(device)
mlp_v1.load_state_dict(torch.load('./best_MPN/mlp_v1.pt', map_location=device))
mlp_v1.eval()

mlp_v2 = MLPRegressor(input_dim=len(feature_cols_v2), hidden_layer_sizes=(128, 64, 16)).to(device)
mlp_v2.load_state_dict(torch.load('./best_MPN/mlp_v2.pt', map_location=device))
mlp_v2.eval()

X_test_v1 = X_test_v1[feature_cols_v1]
X_test_v2 = X_test_v2[feature_cols_v2]

y_pred_v1 = mlp_v1.predict(X_test_v1)
y_pred_v2 = mlp_v2.predict(X_test_v2)

y_pred_v1 = target_scaler_v1.inverse_transform(y_pred_v1)
y_pred_v2 = target_scaler_v2.inverse_transform(y_pred_v2)

test_df["y_pred_v1"] = y_pred_v1
test_df["y_pred_v2"] = y_pred_v2

print("v1")
print("RMSE:", mean_squared_error(test_df["MTB Standardized Residuals"], test_df["y_pred_v1"], squared=False))
print("R²:", r2_score(test_df["MTB Standardized Residuals"], test_df["y_pred_v1"]))
print("\nv2")
print("RMSE:", mean_squared_error(test_df["MTB Standardized Residuals"], test_df["y_pred_v2"], squared=False))
print("R²:", r2_score(test_df["MTB Standardized Residuals"], test_df["y_pred_v2"]))

# test_df.to_csv("./results/test_compound_predictions.csv", index=False)
