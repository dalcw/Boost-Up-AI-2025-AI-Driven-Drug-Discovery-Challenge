# library import
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, MACCSkeys, AllChem, Crippen, QED
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors

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

from gnn_model.graph_feature_extractor import smiles_to_graph_data
from gnn_model.utils import normalize_node_features, normalize_global_features

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def preprocessing(data, node_mean=None, node_std=None, global_mean=None, global_std=None):
    graphs = []
    for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        try: graphs.append(smiles_to_graph_data(row))
        except: pass
    
    dataloader = DataLoader(graphs, batch_size=128, shuffle=True)

    # normalization
    all_node_features = []
    for data in graphs:
        all_node_features.append(data.x.cpu().numpy())
    all_node_features = np.concatenate(all_node_features, axis=0)  # shape (총 노드수, feat_dim)

    if node_mean == None:
        node_mean = all_node_features.mean(axis=0)
        node_std = all_node_features.std(axis=0)
    else:
        pass
    
    all_global_features = []
    for data in graphs:
        all_global_features.append(data.glo_x.cpu().numpy())
    all_global_features = np.stack(all_global_features, axis=0)  # shape (num_graphs, global_feat_dim)

    if global_mean == None:
        global_mean = all_global_features.mean(axis=0)
        global_std = all_global_features.std(axis=0)
    else:
        pass

    for d in graphs:
        d.x = torch.tensor(normalize_node_features(d.x.cpu().numpy(), node_mean, node_std), dtype=torch.float32).to(d.x.device)
        d.glo_x = torch.tensor(normalize_global_features(d.glo_x.cpu().numpy(), global_mean, global_std), dtype=torch.float32).to(d.glo_x.device)

    return dataloader, node_mean, node_std, global_mean, global_std

if __name__ == "__main__":
    data = pd.read_csv("../main_data/train.csv")
    print(preprocessing(data))