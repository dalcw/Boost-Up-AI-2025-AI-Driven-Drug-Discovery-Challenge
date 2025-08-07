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

from h2o_model.feature_extractor import feature_extractor

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')


def inference(model, data):
    encodings = []

    for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        encodings.append(feature_extractor(row["Canonical_Smiles"]))
    
    X = np.array(encodings)
    data = pd.DataFrame(X)
    
    # test file generation
    data_h2o = h2o.H2OFrame(data)
    preds = model.leader.predict(data_h2o)

    return preds