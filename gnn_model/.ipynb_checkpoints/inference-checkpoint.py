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

from gnn_model.utils import *
from gnn_model.model import SmileGIN
from gnn_model.data_preprocessing import preprocessing
from gnn_model.graph_feature_extractor import smiles_to_graph_data
from gnn_model.training import training

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def inference(data, model, node_mean, node_std, global_mean, global_std):
    # test = pd.read_csv("./main_data/test.csv")
    
    graphs = []
    for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        try: graphs.append(smiles_to_graph_data(row, False))
        except: pass
    
    for data in graphs:
        data.x = torch.tensor(normalize_node_features(data.x.cpu().numpy(), node_mean, node_std), dtype=torch.float32).to(data.x.device)
        data.glo_x = torch.tensor(normalize_global_features(data.glo_x.cpu().numpy(), global_mean, global_std), dtype=torch.float32).to(data.glo_x.device)
    
    predicts = []
    model.eval()
    with torch.no_grad():
        for data in graphs:
    
            preds = model(
                    data.x.to(device), 
                    data.glo_x.to(device),
                    data.edge_index.to(device),
                    torch.zeros(data.num_nodes, dtype=torch.long).to(device)
                )
    
            predicts.append(preds.item())
    
    # # submission
    # sub = pd.read_csv("./main_data/sample_submission.csv")
    # sub["Inhibition"] = predicts
    
    # sub.to_csv("./submission/submission_gnn.csv", index=False)

    return predicts


if __name__ == "__main__":
    train = pd.read_csv("../main_data/train.csv")
    test = pd.read_csv("../main_data/test.csv")

    # training
    model = SmileGIN(atom_dim=22, global_dim=15, hidden_dim=64).to(device)
    trainloader, node_mean, node_std, global_mean, global_std = preprocessing(train)

    model = training(model, trainloader)

    # inference
    preds = inference(test, model, node_mean, node_std, global_mean, global_std)
    print(preds)