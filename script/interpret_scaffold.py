"Load the best model to rank the scaffolds by predicted permeability."

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from models.mlp import MLPRegressor
import pandas as pd
import joblib
import torch


def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def rank_scaffolds_by_permeability(df, mlp_optimal, feature_cols, top_n=20):
    # Step 1: Extract scaffolds
    df = df.copy()
    df['scaffold'] = df['Smiles'].apply(get_bemis_murcko_scaffold)
    df = df.dropna(subset=['scaffold'])

    # # Fuse embeddings with descriptors
    # labeled_data_descriptors = pd.read_csv('./data/preprocessed_labeled_descriptors.csv')
    # descriptor_df = labeled_data_descriptors.groupby('Smiles').first().reset_index()
    # df_pro = df.merge(descriptor_df, on='Smiles', how='left')

    # Step 2: Aggregate features by scaffold (mean of features for each scaffold)
    scaffold_features = df.groupby('scaffold')[feature_cols].mean().reset_index()

    # Step 3: Predict permeability
    X = scaffold_features[feature_cols]
    y_pred = mlp_optimal.predict(X)
    scaffold_features['predicted_permeability'] = y_pred

    # Step 4: Rank and return top N
    scaffold_features = scaffold_features.sort_values(by='predicted_permeability').reset_index(drop=True)
    # Only keep scaffold and predicted_permeability
    scaffold_features = scaffold_features[['scaffold', 'predicted_permeability']]
    scaffold_features.to_csv('./results/scaffold_permeability_whole.csv', index=False)
    return scaffold_features.sort_values(by='predicted_permeability').head(top_n).reset_index(drop=True)


train_emb = pd.read_csv('./model/train_emb.csv')
val_emb = pd.read_csv('./model/val_emb.csv')
test_emb = pd.read_csv('./model/test_emb.csv')
df_emb_best = pd.concat([train_emb, val_emb, test_emb], axis=0, ignore_index=True)

feature_cols = joblib.load('./model/mlp_feature_cols.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_optimal = MLPRegressor(input_dim=len(feature_cols), hidden_layer_sizes=(128, 64, 16)).to(device)
mlp_optimal.load_state_dict(torch.load('./model/optimal_mlp.pt'))
mlp_optimal.eval()

top20 = rank_scaffolds_by_permeability(df_emb_best, mlp_optimal, feature_cols)
print(top20)
