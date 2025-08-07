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

from gnn_model.feature_extractor import atom_to_feature, global_feature_extractor

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')
device = "cuda:0" if torch.cuda.is_available() else "cpu"



def smiles_to_graph_data(row, train=True):
    mol = Chem.MolFromSmiles(row["Canonical_Smiles"])
    if mol is None:
        raise ValueError("Invalid SMILES")

    # 노드 피처
    x = torch.tensor([atom_to_feature(atom) for atom in mol.GetAtoms()], dtype=torch.float)
        
    # 엣지 정보
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    # 정답값
    if train:
        y = torch.tensor(row["Inhibition"], dtype=torch.float)
    else:
        y = 0

    # 전역 피처
    global_x = global_feature_extractor(mol)
    global_x = torch.tensor([global_feature_extractor(mol)], dtype=torch.float)

    return Data(x=x, glo_x=global_x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    train = pd.read_csv("../main_data/train.csv")
    print(smiles_to_graph_data(train.iloc[0]))