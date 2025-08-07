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

from gnn_model.model import SmileGIN
from gnn_model.data_preprocessing import preprocessing
from gnn_model.utils import mixup_embeddings

# 모든 경고 끄기
# RDLogger.DisableLog('rdApp.*')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def training(model, dataloader):
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    # epoch
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
    
        for data in tqdm.tqdm(dataloader):
            # data
            graph_x = data.x.to(device)
            global_x = data.glo_x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)    
            y = data.y.to(device)
    
            embeddings = model.embedding_layer(graph_x, global_x, edge_index, batch)
    
            # mixup
            embeddings_mix, y_mix = mixup_embeddings(embeddings, y)
    
            # prediction layer
            predict = model.regression_layer(embeddings_mix)
    
            # parameter update
            optimizer.zero_grad()
    
            loss = F.mse_loss(predict.squeeze(-1), y_mix.to(torch.float))
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        # scheduler.step()
        print(f"[Train Loss - Epoch: {epoch+1}]\t{total_loss / len(dataloader):.5f}")

    return model


if __name__ == "__main__":
    model = SmileGIN(atom_dim=22, global_dim=15, hidden_dim=64).to(device)

    data = pd.read_csv("../main_data/train.csv")
    dataloader, _, _, _, _ = preprocessing(data)

    training(model, dataloader)
