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

# automl library
import h2o
from h2o.automl import H2OAutoML

import numpy as np
import pandas as pd
import tqdm
import argparse

from h2o_model.data_preprocessing import processing as h2o_processing
from h2o_model.training import training as h2o_training
from h2o_model.inference import inference as h2o_inference

from gnn_model.utils import *
from gnn_model.model import SmileGIN
from gnn_model.data_preprocessing import preprocessing as gnn_processing
from gnn_model.graph_feature_extractor import smiles_to_graph_data
from gnn_model.training import training as gnn_training
from gnn_model.inference import inference as gnn_inference


# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2o_sec", type=int, default=3600)
    args = parser.parse_args()
    
    
    # main dataset
    main_data = pd.read_csv("./main_data/train.csv")
    
    # AID_1851
    aid_1851_data = pd.read_csv("./main_data/AID_1851_datatable_all.csv")
    # cyp3a4만을 대상으로 하는 데이터만을 필터링
    aid_1851_data = aid_1851_data[aid_1851_data["Panel Name"] == "p450-cyp3a4"]
    # smiles와 11.4um에서 실험된 activity 값을 추출
    aid_1851_data = aid_1851_data[["PUBCHEM_EXT_DATASOURCE_SMILES", "Activity at 11.43 uM"]]
    # activation 값에 음수를 취함 (이게 저해율이라고 판단함)
    aid_1851_data["Activity at 11.43 uM"] = aid_1851_data.apply(lambda x: -float(x["Activity at 11.43 uM"]), axis=1)
    # 칼럼 이름 변경
    aid_1851_data.columns = ["Canonical_Smiles", "Inhibition"]
    
    # AID 884
    # AID 1851과 동일한 방식의 전처리로 처리
    aid_884_data = pd.read_csv("./main_data/AID_884_datatable_all.csv")
    aid_884_data = aid_884_data[["PUBCHEM_EXT_DATASOURCE_SMILES", "Activity at 11.43 uM"]]
    aid_884_data.dropna(inplace=True)
    aid_884_data["Activity at 11.43 uM"] = aid_884_data.apply(lambda x: -float(x["Activity at 11.43 uM"]), axis=1)
    aid_884_data.columns = ["Canonical_Smiles", "Inhibition"]
    
    # Full data
    train = pd.concat([main_data, aid_1851_data, aid_884_data], axis=0)
    
    # 만약 동일한 데이터가 존재할 경우에는 평균으로 aggregation 함
    train = train.groupby("Canonical_Smiles", as_index=False).agg({"Inhibition": "mean"})
    # Inhibition의 값이 0 이상인 경우의 데이터만 유효하다고 판단함
    train = train[train["Inhibition"] >= 0]

    # test file
    test = pd.read_csv("./main_data/test.csv")


    h2o learning
    print("[+] H2O model training")
    h2o_processed_train = h2o_processing(train)
    h2o_model = h2o_training(h2o_processed_train, args.h2o_sec)
    h2o_predict = h2o_inference(h2o_model, test).as_data_frame().values.reshape(-1).copy()
    # print(h2o_predict)
    h2o.shutdown(prompt=False)

    # gnn learning
    print("[+] GNN model training")
    gnn_model = SmileGIN(atom_dim=22, global_dim=15, hidden_dim=64).to(device)
    trainloader, node_mean, node_std, global_mean, global_std = gnn_processing(main_data)
    gnn_model = gnn_training(gnn_model, trainloader)
    torch.save(gnn_model.state_dict(), "./model_weight/gnn.pt")

    gnn_predict = np.array(gnn_inference(test, gnn_model, node_mean, node_std, global_mean, global_std))
    # print(gnn_predict)

    # ensemble
    final_predict = 0.9 * h2o_predict + 0.1 * gnn_predict

    sub = pd.read_csv("./main_data/sample_submission.csv")
    sub["Inhibition"] = final_predict
    
    sub.to_csv("./submission/submission.csv", index=False)
    print("[+] Generate [submission file]")
