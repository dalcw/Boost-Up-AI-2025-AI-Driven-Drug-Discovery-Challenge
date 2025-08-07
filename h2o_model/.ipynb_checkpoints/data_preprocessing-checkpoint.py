import numpy as np
import pandas as pd
import tqdm
from h2o_model.feature_extractor import feature_extractor

def processing(train):
    train_encodings = []

    for _, row in tqdm.tqdm(train.iterrows(), total=len(train)):
        train_encodings.append(feature_extractor(row["Canonical_Smiles"]))
    
    train_X = np.array(train_encodings)
    train_y = np.array(train["Inhibition"])
    
    train = np.concatenate([train_X, train_y.reshape(-1, 1)], axis=1)
    train = pd.DataFrame(train)

    return train

if __name__ == "__main__":
    train = pd.read_csv("../main_data/train.csv")
    print(processing(train).shape)