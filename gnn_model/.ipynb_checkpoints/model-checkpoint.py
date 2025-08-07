import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LayerNorm, Dropout
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, GINConv, AttentionalAggregation
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
import tqdm

from gnn_model.data_preprocessing import preprocessing

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class SmileGIN(nn.Module):
    def __init__(self, atom_dim, global_dim, hidden_dim, dropout=0):
        super(SmileGIN, self).__init__()

        # embedding block: graph + global
        self.backbone = nn.ModuleDict({
            "atom_proj": Linear(atom_dim, hidden_dim),
            "gin_layers": nn.ModuleList([
                GINConv(Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                )) for _ in range(2)
            ]),
            "norm": LayerNorm(hidden_dim),
            "pool": AttentionalAggregation(
                gate_nn=Linear(hidden_dim, 1),
                nn=Sequential(Linear(hidden_dim, hidden_dim), ReLU())
            ),
            "global_embed": nn.Sequential(
                Linear(global_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU(),
            )
        })

        self.embedding = Linear(hidden_dim * 3, 64)

        # head block: fusion + regression
        self.head = nn.Sequential(
            Linear(64, 64),
            nn.ReLU(),
            Linear(64, 1)
        )

    def embedding_layer(self, graph_x, global_x, edge_index, batch):
        # embedding: atom features → GIN → pooling
        x = F.relu(self.backbone["atom_proj"](graph_x))
        for gin in self.backbone["gin_layers"]:
            x = x + gin(x, edge_index)
        x = self.backbone["norm"](x)
        graph_feat = self.backbone["pool"](x, batch)

        # embedding: global features
        global_feat = self.backbone["global_embed"](global_x)

        # fusion
        fusion = graph_feat * global_feat
        merged = torch.cat([graph_feat, global_feat, fusion], dim=1)
        embedding = self.embedding(merged)
        return embedding

    def regression_layer(self, embedding):
        return self.head(embedding)

    def forward(self, graph_x, global_x, edge_index, batch):
        embedding = self.embedding_layer(graph_x, global_x, edge_index, batch)
        output = self.regression_layer(embedding)
        return output


if __name__ == "__main__":
    model = SmileGIN(atom_dim=22, global_dim=15, hidden_dim=64).to(device)

    data = pd.read_csv("../main_data/train.csv")
    train_graphs, node_mean, node_std, global_mean, global_std = preprocessing(data)
    
    result = model(
                train_graphs[0].x.to(device),
                train_graphs[0].glo_x.to(device),
                train_graphs[0].edge_index.to(device),
                torch.zeros(train_graphs[0].num_nodes, dtype=torch.long).to(device)
            )
    print(result)