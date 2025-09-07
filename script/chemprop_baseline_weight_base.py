"MINE weight search — baseline performance."

from data_tools.pyg_chemprop_utils import initialize_weights, directed_mp, aggregate_at_nodes, NoamLR, get_dataset, get_dataloaders, get_emb_dataloaders
from typing import Any, Dict, Optional
from models.mlp import MLPRegressor

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import copy
import random
import itertools

import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, ModuleList, Dropout, MSELoss

from torch_geometric.nn import GINEConv, global_mean_pool, GPSConv, global_add_pool

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
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
        self.bn_1 = BatchNorm1d(hidden_size)
        self.bn_2 = BatchNorm1d(hidden_size)
        self.bn_3 = BatchNorm1d(hidden_size)
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

        return self.mlp(x)

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


def train(model, train_loader, optimizer, scheduler):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # out = model(data.x, data.pe, data.edge_index, data.edge_attr,
        #             data.batch)
        out = model(data)
        # loss = (out.squeeze() - data.y).abs().mean()
        criterion = MSELoss()
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        # out = model(data.x, data.pe, data.edge_index, data.edge_attr,
        #             data.batch)
        out = model(data)
        # total_error += (out.squeeze() - data.y).abs().sum().item()
        criterion = MSELoss()
        loss = criterion(out.squeeze(), data.y.squeeze())
        total_loss += loss.item() * data.num_graphs
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
def get_representations(model, loader, scaler=None, smile=False):
    model.eval()
    smiles_list = []
    embeddings = []
    y_list = []

    for data in tqdm(loader):
        data = data.to(device)

        # Forward to get representations
        rep = model.representation(data)  # shape: (batch_size, dim)
        embeddings.append(rep.cpu().numpy())

        # Extract and inverse target
        y = data.y.cpu().numpy()
        if scaler is not None:
            y = scaler.inverse_transform(y)
        y_list.append(y)

        # Assumes `data.smiles` is a list of SMILES strings of length batch_size
        smiles_batch = data.smiles if isinstance(data.smiles, list) else list(data.smiles)
        smiles_list.extend(smiles_batch)

    # Stack all embeddings and labels
    embeddings = np.concatenate(embeddings)  # shape: (N, emb_dim)
    y = np.concatenate(y_list)               # shape: (N,)

    df_embed = pd.DataFrame(embeddings, columns=[f'ft_{i}' for i in range(embeddings.shape[1])])

    if smile:
        df_embed.insert(0, "Smiles", smiles_list)  # insert at column 0

    return df_embed, y


# Seed combinations
all_seeds = list(range(42, 92))
random.seed(42)  # For reproducibility
torch_seeds = random.sample(all_seeds, 3)
data_seeds = random.sample(all_seeds, 3)

print("Torch Seeds:", torch_seeds)
print("Data Seeds:", data_seeds)

mlp_seeds = random.sample(range(100, 200), 9)
seed_combinations = list(itertools.product(torch_seeds, data_seeds))

results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = './results/chemprop_permeability_baseline_weight_checkpoint'
os.makedirs(save_dir, exist_ok=True)

train_df = pd.read_csv('./data/train_scaffold_split.csv')
test_df = pd.read_csv('./data/test_scaffold_split.csv')

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

    model = DMPNNEncoder(
        hidden_size=300, node_fdim=133, edge_fdim=14,
        depth=3, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    scheduler = NoamLR(
        optimizer,
        warmup_epochs=[3],
        total_epochs=[30],
        steps_per_epoch=len(train_loader),
        init_lr=[1e-4],
        max_lr=[1e-3],
        final_lr=[1e-4],
    )

    best_val = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in tqdm(range(1, 31)):
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

        # if patience_counter >= patience:
        #     print(f"Stop at epoch {epoch - patience_counter} with the best val loss: {best_val}")
        #     break

    print(f"Best validation loss: {best_val} | RMSE: {np.sqrt(best_val)}")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))

    train_loader, val_loader, test_loader = get_emb_dataloaders(
        train_val_dataset=train_val_dataset_orig,
        test_dataset=test_dataset_orig,
        seed=mlp_seed,
    )

    X_train, y_train = get_representations(model, train_loader)
    X_val, y_val = get_representations(model, val_loader)
    X_test, y_test = get_representations(model, test_loader)

    if i == 0:
        print("X_train shape:", X_train.shape)
        print(X_train.iloc[0])
        print(y_train[0])

    model = MLPRegressor(input_dim=X_train.shape[1], hidden_layer_sizes=(128, 64, 16))
    mlp_optimal = model.fit(X_train, y_train, X_val, y_val,
                            alpha=0.01, batch_size=64, learning_rate_init=0.0005,
                            random_state=mlp_seed, early_stopping=True,
                            patience=10, max_epochs=100)

    if i == 0:
        print(model)

    y_val_pred = mlp_optimal.predict(X_val)
    y_test_pred = mlp_optimal.predict(X_test)
    results.append({
        'Torch Seed': torch_seed,
        'Data Seed': data_seed,
        'MLP Seed': mlp_seed,
        'Validation R2': r2_score(y_val, y_val_pred),
        'Validation RMSE': root_mean_squared_error(y_val, y_val_pred),
        'Validation Spearman': spearmanr(y_val, y_val_pred).correlation,
        'Test R2': r2_score(y_test, y_test_pred),
        'Test RMSE': root_mean_squared_error(y_test, y_test_pred),
        'Test Spearman': spearmanr(y_test, y_test_pred).correlation,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, 'chemprop_baseline_weight_performance.csv'), index=False)
