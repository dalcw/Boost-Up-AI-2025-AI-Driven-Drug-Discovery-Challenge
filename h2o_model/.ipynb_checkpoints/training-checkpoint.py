# automl library
import h2o
from h2o.automl import H2OAutoML

# chemical feature extractor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, MACCSkeys, AllChem, Crippen
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# basic library
import numpy as np
import pandas as pd
import tqdm

from h2o_model.data_preprocessing import processing

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')

def training(data, runtime_sec, target_idx=2236, ):
    h2o.init()
    
    # train data
    train_h2o = h2o.H2OFrame(data)
    
    # target column (last column)
    target = target_idx
    features = [col for col in train_h2o.columns if col != target]
    
    train_h2o[target] = train_h2o[target].asnumeric()
    
    # AutoML 수행
    aml = H2OAutoML(max_runtime_secs=runtime_sec, sort_metric="RMSE")
    aml.train(x=features, y=target, training_frame=train_h2o)

    return aml

if __name__ == "__main__":
    train = pd.read_csv("../main_data/train.csv")
    processed_train = processing(train)
    training(processed_train, 300)