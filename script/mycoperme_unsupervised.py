"Unsupervised pretraining by maximizing mutual information between molecular graph and descriptors."

from data_tools.pyg_chemprop_utils import (initialize_weights, directed_mp,
                                           aggregate_at_nodes, NoamLR, get_dataset,
                                           get_dataloaders)
from models import MINE
from models.GNNs import AttentiveFP, GINE, GCN

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import random
import argparse

import torch
from torch.nn import Linear, ReLU, Sequential, Dropout

from torch_geometric.nn import global_mean_pool


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
          mine=None):
    model.train()
    mine.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, emb = model(data)
        loss = mine(emb, data.fg.to(device))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, mine=None):
    model.eval()
    mine.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, emb = model(data)
        loss = mine(emb, data.fg.to(device))
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


parser = argparse.ArgumentParser(description='MoleculeNet benckmark')
parser.add_argument('--GNN', type=str, default='chemprop', help='First-stage GNN')
args = parser.parse_args()

GNN = args.GNN

# Seed combinations
all_seeds = list(range(42, 92))
random.seed(42)  # For reproducibility
seeds = random.sample(all_seeds, 3)

print("Random Seeds:", seeds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = f'./results/{GNN}_permeability_unsupervised_checkpoint'
os.makedirs(save_dir, exist_ok=True)

train_df = pd.read_csv('./data/unsupervised_descriptors.csv')
test_df = pd.read_csv('./data/test_scaffold_split.csv')

smiles_to_fingerprint = {row['Smiles']: row.iloc[3:].astype(float).values for _, row in train_df.iterrows()}

labeled_data_descriptors = pd.read_csv('./data/preprocessed_labeled_descriptors.csv')
smiles_to_fingerprint_test = {row['Smiles']: row.iloc[1:].astype(float).values for _, row in labeled_data_descriptors.iterrows()}


for i, seed in enumerate(seeds):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    if i == 0:
        train_val_dataset = get_dataset(train_df, smiles_to_fingerprint)
        test_dataset = get_dataset(test_df, smiles_to_fingerprint_test)
        torch.save(train_val_dataset, os.path.join(save_dir, 'train_val_dataset.pt'))
        torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
    else:
        train_val_dataset = torch.load(os.path.join(save_dir, 'train_val_dataset.pt'))
        test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pt'))

    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        batch_size=2048,
        seed=seed,
        generator=g
    )

    attn_kwargs = {'dropout': 0.0}
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

    mine = MINE(x_dim=300, y_dim=184, hidden_size=512).to(device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(mine.parameters()),
                                 lr=0.0001, weight_decay=0)
    scheduler = NoamLR(
        optimizer,
        warmup_epochs=[10],
        total_epochs=[100],
        steps_per_epoch=len(train_loader),
        init_lr=[1e-4],
        max_lr=[1e-3],
        final_lr=[1e-4],
    )

    best_val = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in tqdm(range(1, 101)):
        loss = train(model, train_loader, optimizer, scheduler, mine)
        val_mae = test(model, val_loader, mine)
        test_mae = test(model, test_loader, mine)

        if val_mae < best_val:
            best_val = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_{i}.pt'))
        else:
            patience_counter += 1

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}')

        if patience_counter >= patience:
            print(f"Stop at epoch {epoch - patience_counter} with the best val loss: {best_val}")
            break

    print(f"Best validation loss: {best_val}")
