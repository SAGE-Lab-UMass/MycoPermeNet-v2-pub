"NST and Fusion+NST with the unlabeled In-house enamine dataset."

from data_tools.pyg_chemprop_utils import (initialize_weights, directed_mp,
                                           aggregate_at_nodes, NoamLR, get_dataset,
                                           get_dataloaders, get_emb_dataloaders,
                                           get_dataset_from_mlsmr, get_dataloaders_from_mlsmr,
                                           get_dataset_from_enamine)
from models.mlp import MLPRegressor
from models.GNNs import AttentiveFP, GINE, GCN

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import copy
import random
import itertools
import argparse

import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, MSELoss

from torch_geometric.nn import global_mean_pool

from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import spearmanr


def set_seed(data_seed, torch_seed):
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


class DMPNNEncoder(torch.nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3, dropout=0):
        super(DMPNNEncoder, self).__init__()
        self.act_func = ReLU()
        self.W1 = Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = Linear(hidden_size, hidden_size, bias=False)
        self.W3 = Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth

        self.mlp = Sequential(
            Dropout(p=dropout, inplace=False),
            Linear(hidden_size, hidden_size, bias=True),
            ReLU(),
            Dropout(p=dropout, inplace=False),
            Linear(hidden_size, 1, bias=True),
        )

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        x = global_mean_pool(node_attr, batch)

        return self.mlp(x), x

    def representation(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        return global_mean_pool(node_attr, batch)


def train(model, train_loader, optimizer, scheduler,
          mine=None, combined_weight=None,):
    model.train()
    if use_MINE:
        mine.train()

    total_loss = 0
    total_mi_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, emb = model(data)
        criterion = MSELoss()
        if use_MINE:
            mi_loss = mine(emb, data.fg.to(device))
            loss = criterion(out.squeeze(), data.y.squeeze()) + combined_weight * mi_loss
        else:
            loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        if use_MINE:
            total_mi_loss += mi_loss.item() * data.num_graphs
        optimizer.step()
        scheduler.step()
    if use_MINE:
        return total_loss / len(train_loader.dataset), total_mi_loss / len(train_loader.dataset)
    else:
        return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, mine=None, combined_weight=None):
    model.eval()
    if use_MINE:
        mine.eval()

    total_loss = 0
    total_mi_loss = 0
    for data in loader:
        data = data.to(device)
        out, emb = model(data)
        criterion = MSELoss()
        if use_MINE:
            mi_loss = mine(emb, data.fg.to(device))
            loss = criterion(out.squeeze(), data.y.squeeze()) + combined_weight * mi_loss
        else:
            loss = criterion(out.squeeze(), data.y.squeeze())
        total_loss += loss.item() * data.num_graphs
        if use_MINE:
            total_mi_loss += mi_loss.item() * data.num_graphs
    if use_MINE:
        return total_loss / len(loader.dataset), total_mi_loss / len(loader.dataset)
    else:
        return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    print("Size of the loader:", len(loader.dataset))
    for data in loader:
        data = data.to(device)
        # out = model(data.x, data.pe, data.edge_index, data.edge_attr,
        #             data.batch)
        out = model(data)
        all_preds.append(out.cpu().numpy())
        all_targets.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    rmse = root_mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    spearman_corr, _ = spearmanr(all_targets, all_preds)

    return r2, rmse, spearman_corr


@torch.no_grad()
def get_representations(model, loader, NST=False, scaler=None, smile=False, fusion_nst=False):
    model.eval()
    smiles_list = []
    embeddings = []
    y_list = []
    desc_list = []

    for data in tqdm(loader):
        data = data.to(device)

        # Forward to get representations
        rep = model.representation(data)  # shape: (batch_size, dim)
        embeddings.append(rep.cpu().numpy())

        if not NST:
            # Extract and inverse target
            y = data.y.cpu().numpy()
            if scaler is not None:
                y = scaler.inverse_transform(y)
            y_list.append(y)

        # (optional) get descriptor vector for fusion
        if fusion_nst and hasattr(data, 'rdkit'):
            desc = data.rdkit.cpu().numpy()  # shape: (B, D2)
            desc_list.append(desc)

        # Assumes `data.smiles` is a list of SMILES strings of length batch_size
        smiles_batch = data.smiles if isinstance(data.smiles, list) else list(data.smiles)
        smiles_list.extend(smiles_batch)

    # Stack all embeddings and labels
    embeddings = np.concatenate(embeddings)  # shape: (N, emb_dim)
    if not NST:
        y = np.concatenate(y_list)               # shape: (N,)

    df_embed = pd.DataFrame(embeddings, columns=[f'ft_{i}' for i in range(embeddings.shape[1])])

    if smile:
        df_embed.insert(0, "Smiles", smiles_list)  # insert at column 0

    if fusion_nst:
        X_desc = np.concatenate(desc_list)  # (N, D2)
        X = np.concatenate([embeddings, X_desc], axis=1)
        embed_columns = [f'ft_{i}' for i in range(embeddings.shape[1])]
        desc_columns = descriptor_names
        df_embed = pd.DataFrame(X, columns=embed_columns + desc_columns)

    if not NST:
        return df_embed, y
    else:
        return df_embed


parser = argparse.ArgumentParser(description='MoleculeNet benckmark')
parser.add_argument('--GNN', type=str, default='chemprop', help='First-stage GNN')
parser.add_argument('--fusion', type=bool, default=False, help='Fuse RDKit descriptors')
parser.add_argument('--NST', type=bool, default=False, help='semi-supervised noisy student self-distillation')
parser.add_argument('--use_MINE', type=bool, default=False, help='Use MINE loss')
parser.add_argument('--NST_volume', type=int, default=1000, help='Number of unlabeled data for each NST iteration')
args = parser.parse_args()

GNN = args.GNN
fusion = args.fusion
NST = args.NST
use_MINE = args.use_MINE
NST_volume = args.NST_volume

# Seed combinations
all_seeds = list(range(42, 92))
random.seed(42)  # For reproducibility
torch_seeds = random.sample(all_seeds, 7)
data_seeds = random.sample(all_seeds, 7)

print("Torch Seeds:", torch_seeds)
print("Data Seeds:", data_seeds)

mlp_seeds = random.sample(range(100, 200), 49)
seed_combinations = list(itertools.product(torch_seeds, data_seeds))

results = []
val_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if fusion and use_MINE:
    save_dir = f'./results/{GNN}_permeability_mine_fusion_checkpoint'
elif fusion and NST:
    save_dir = f'./results/{GNN}_permeability_enamine_fusion_nst_checkpoint'
elif fusion:
    save_dir = f'./results/{GNN}_permeability_fusion_checkpoint'
elif use_MINE:
    save_dir = f'./results/{GNN}_permeability_mine_checkpoint'
elif NST:
    save_dir = f'./results/{GNN}_permeability_enamine_nst_checkpoint'
else:
    save_dir = f'./results/{GNN}_permeability_checkpoint'
os.makedirs(save_dir, exist_ok=True)

train_df = pd.read_csv('./data/train_scaffold_split.csv')
test_df = pd.read_csv('./data/test_scaffold_split.csv')

labeled_data_descriptors = pd.read_csv('./data/preprocessed_labeled_descriptors.csv')
descriptor_names = labeled_data_descriptors.columns.drop("Smiles").tolist()

# Load all unique unlabeled SMILES once
df_unlabeled = pd.read_csv('./data/enamine.csv')
unique_unlabeled_smiles = df_unlabeled["SMILES"].drop_duplicates().tolist()

nst_iterations = 3
total_required = nst_iterations * NST_volume
assert len(unique_unlabeled_smiles) >= total_required, "Not enough unique SMILES for all iterations without overlap."


for i, (torch_seed, data_seed) in enumerate(seed_combinations):
    mlp_seed = mlp_seeds[i]
    print(f"Running experiment with Torch seed {torch_seed}, Data seed {data_seed}, MLP seed {mlp_seed}")

    set_seed(data_seed, torch_seed)
    g = torch.Generator()
    g.manual_seed(torch_seed)

    if i == 0:
        train_val_dataset = get_dataset(train_df)
        test_dataset = get_dataset(test_df)
        torch.save(train_val_dataset, os.path.join(save_dir, 'train_val_dataset.pt'))
        torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
    else:
        train_val_dataset = torch.load(os.path.join(save_dir, 'train_val_dataset.pt'))
        test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pt'))

    train_val_dataset_orig = copy.deepcopy(train_val_dataset)
    test_dataset_orig = copy.deepcopy(test_dataset)

    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        batch_size=50,
        seed=data_seed,
        generator=g
    )

    attn_kwargs = {'dropout': 0.0}
    # model = GPS(channels=300, pe_dim=20, num_layers=6, num_heads=5,
    #             attn_type='multihead', attn_kwargs=attn_kwargs).to(device)
    if GNN == 'chemprop':
        model = DMPNNEncoder(
            hidden_size=300, node_fdim=133, edge_fdim=14,
            depth=3, dropout=0).to(device)
    elif GNN == 'AttentiveFP':
        model = AttentiveFP(
            in_channels=133, hidden_channels=300, out_channels=1,
            edge_dim=14, num_layers=5, num_timesteps=2, dropout=0.3).to(device)
    elif GNN == 'GINE':
        model = GINE(
            in_channels=133, hidden_channels=300, num_layers=5, out_channels=1,
            dropout=0.2, jk='last', edge_dim=14).to(device)
    elif GNN == 'GCN':
        model = GCN(
            in_channels=133, hidden_channels=300, num_layers=5, out_channels=1,
            dropout=0.2, jk='last').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    scheduler = NoamLR(
        optimizer,
        warmup_epochs=[3],
        total_epochs=[50],
        steps_per_epoch=len(train_loader),
        init_lr=[1e-4],
        max_lr=[1e-3],
        final_lr=[1e-4],
    )

    best_val = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in tqdm(range(1, 51)):
        loss = train(model, train_loader, optimizer, scheduler)
        val_mae = test(model, val_loader)
        test_mae = test(model, test_loader)

        if val_mae < best_val:
            best_val = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}')

        if patience_counter >= patience:
            print(f"Stop at epoch {epoch - patience_counter} with the best val loss: {best_val}")
            break

    print(f"Best validation loss: {best_val} | RMSE: {np.sqrt(best_val)}")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))

    train_loader, val_loader, test_loader = get_emb_dataloaders(
        train_val_dataset=train_val_dataset_orig,
        test_dataset=test_dataset_orig,
        seed=mlp_seed,
    )

    X_train, y_train = get_representations(model, train_loader, smile=fusion)
    X_val, y_val = get_representations(model, val_loader, smile=fusion)
    X_test, y_test = get_representations(model, test_loader, smile=fusion)

    if fusion:
        # Fusion concatenation
        descriptor_df = labeled_data_descriptors.groupby('Smiles').first().reset_index()

        X_train = X_train.merge(descriptor_df, on='Smiles', how='left')
        X_val = X_val.merge(descriptor_df, on='Smiles', how='left')
        X_test = X_test.merge(descriptor_df, on='Smiles', how='left')

        X_train = X_train.drop(columns=['Smiles'])
        X_val = X_val.drop(columns=['Smiles'])
        X_test = X_test.drop(columns=['Smiles'])

    if i == 0:
        print("X_train shape:", X_train.shape)
        print(X_train.iloc[0])
        print(y_train[0])

    mlp_model = MLPRegressor(input_dim=X_train.shape[1], hidden_layer_sizes=(128, 64, 16))
    mlp_optimal = mlp_model.fit(X_train, y_train, X_val, y_val,
                            alpha=0.01, batch_size=64, learning_rate_init=0.0005,
                            random_state=mlp_seed, early_stopping=True,
                            patience=10, max_epochs=100)

    if i == 0:
        print(mlp_model)

    val_preds = {}
    test_preds = {}

    y_val_pred = mlp_optimal.predict(X_val)
    y_test_pred = mlp_optimal.predict(X_test)
    val_preds[0] = y_val_pred
    test_preds[0] = y_test_pred

    print(f'Val R2: {r2_score(y_val, y_val_pred)} | RMSE: {root_mean_squared_error(y_val, y_val_pred)}')
    print(f'Test R2: {r2_score(y_test, y_test_pred)} | RMSE: {root_mean_squared_error(y_test, y_test_pred)}')

    if not NST:
        val_results.append({
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
            'R2': r2_score(y_val, y_val_pred),
            'RMSE': root_mean_squared_error(y_val, y_val_pred),
            'Spearman': spearmanr(y_val, y_val_pred).correlation,
        })

        results.append({
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
            'R2': r2_score(y_test, y_test_pred),
            'RMSE': root_mean_squared_error(y_test, y_test_pred),
            'Spearman': spearmanr(y_test, y_test_pred).correlation,
        })

    else:
        # NST iterations
        # Get three non-overlapping partitions of N unique SMILES each iteration
        if i == 0:
            rng = np.random.default_rng(mlp_seed)
            shuffled_smiles = rng.permutation(unique_unlabeled_smiles)
            partitioned_smiles = np.array_split(shuffled_smiles[:total_required], nst_iterations)

        X_nst = X_train.copy()
        y_nst = y_train.copy()
        # mlp_state_dict = mlp_optimal.state_dict()

        for iter_i in range(1, nst_iterations + 1):
            mlp_nst = copy.deepcopy(mlp_optimal)
            if i == 0:
                smiles_chunk = partitioned_smiles[iter_i - 1].tolist()
                if fusion:
                    unlabeled_dataset = get_dataset_from_enamine(smiles_chunk, descriptor_names=descriptor_names)
                else:
                    unlabeled_dataset = get_dataset_from_enamine(smiles_chunk)
                torch.save(unlabeled_dataset, os.path.join(save_dir, f'unlabeled_dataset_{iter_i}.pt'))
            else:
                unlabeled_dataset = torch.load(os.path.join(save_dir, f'unlabeled_dataset_{iter_i}.pt'))

            unlabeled_loader = get_dataloaders_from_mlsmr(unlabeled_dataset)
            X_unlabeled = get_representations(model, unlabeled_loader, NST=NST, fusion_nst=fusion)
            X_unlabeled = X_unlabeled[X_nst.columns]

            # Predict pseudo-labels for unlabeled data
            # y_unlabeled_pred = mlp_optimal.predict(X_unlabeled)
            y_unlabeled_pred = mlp_nst.predict(X_unlabeled)
            np.random.seed(mlp_seed + iter_i)
            y_unlabeled_pred += np.random.normal(0, 0.01, size=y_unlabeled_pred.shape)

            # Combine real + pseudo data
            X_nst = pd.concat([X_nst, X_unlabeled], axis=0)
            y_nst = np.concatenate([y_nst, y_unlabeled_pred], axis=0)

            # Initialize new MLP and load previous weights
            # mlp_nst = MLPRegressor(input_dim=X_nst.shape[1], hidden_layer_sizes=(128, 64, 16))
            # mlp_nst.load_state_dict(mlp_state_dict)
            mlp_optimal = mlp_nst.fit(X_nst, y_nst, X_val, y_val,
                                      alpha=0.01, batch_size=64, learning_rate_init=0.0005,
                                      random_state=mlp_seed, early_stopping=True,
                                      patience=10, max_epochs=100)
            mlp_state_dict = mlp_optimal.state_dict()

            # Save predictions
            y_val_pred = mlp_optimal.predict(X_val)
            y_test_pred = mlp_optimal.predict(X_test)
            val_preds[iter_i] = y_val_pred
            test_preds[iter_i] = y_test_pred

            print(f"NST Iteration {iter_i}")
            print(X_nst.shape)
            print(f'Val R2: {r2_score(y_val, y_val_pred)} | RMSE: {root_mean_squared_error(y_val, y_val_pred)}')
            print(f'Test R2: {r2_score(y_test, y_test_pred)} | RMSE: {root_mean_squared_error(y_test, y_test_pred)}')

        val_result_entry = {
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
        }
        for iter_i in range(nst_iterations + 1):
            val_result_entry[f'Iter{iter_i} R2'] = r2_score(y_val, val_preds[iter_i])
            val_result_entry[f'Iter{iter_i} RMSE'] = root_mean_squared_error(y_val, val_preds[iter_i])
            val_result_entry[f'Iter{iter_i} Spearman'] = spearmanr(y_val, val_preds[iter_i]).correlation
        val_results.append(val_result_entry)

        result_entry = {
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
        }
        for iter_i in range(nst_iterations + 1):
            result_entry[f'Iter{iter_i} R2'] = r2_score(y_test, test_preds[iter_i])
            result_entry[f'Iter{iter_i} RMSE'] = root_mean_squared_error(y_test, test_preds[iter_i])
            result_entry[f'Iter{iter_i} Spearman'] = spearmanr(y_test, test_preds[iter_i]).correlation
        results.append(result_entry)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, f'{GNN}_test_performance.csv'), index=False)

val_results_df = pd.DataFrame(val_results)
val_results_df.to_csv(os.path.join(save_dir, f'{GNN}_val_performance.csv'), index=False)
