# library import
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, MACCSkeys, AllChem, Crippen, QED
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
import pandas as pd
import tqdm

# 모든 경고 끄기
RDLogger.DisableLog('rdApp.*')

def atom_to_feature(atom):
    g_charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
    return [
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()),
        int(atom.GetHybridization() == Chem.HybridizationType.SP),
        int(atom.GetHybridization() == Chem.HybridizationType.SP2),
        int(atom.GetHybridization() == Chem.HybridizationType.SP3),
        int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
        int(atom.GetHybridization() == Chem.HybridizationType.SP3D2),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
        atom.GetDegree(),
        atom.GetImplicitValence(),
        atom.GetNumExplicitHs(),
        atom.GetNumImplicitHs(),
        atom.GetMass(),
        int(atom.GetIsotope()),
        int(atom.GetChiralTag()),
        int(atom.GetNoImplicit()),
        int(atom.HasProp("_CIPCode")),
        g_charge,  # Gasteiger 전하
        atom.GetNumRadicalElectrons(),  # 라디칼 전자 수
        atom.GetTotalValence()
    ]


def global_feature_extractor(mol):
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        rdMolDescriptors.CalcLabuteASA(mol),                       # 접근 가능 표면적
        Descriptors.MolMR(mol),                                    # 몰 굴절률
        rdMolDescriptors.CalcExactMolWt(mol),                      # 정밀 분자량
        Descriptors.NumValenceElectrons(mol),                      # 원자가 전자 수
        len([a for a in mol.GetAtoms() if a.GetSymbol() == 'P'])   # 인(P) 원자 수
    ]

if __name__ == "__main__":
    
    # SMILES 입력
    smiles = "CCO"  # 에탄올 예시
    
    # Mol 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # Gasteiger 전하 계산 (atom_to_feature 내부에서 사용됨)
    AllChem.ComputeGasteigerCharges(mol)
    
    # 원자 단위 feature 추출
    atom_features = [atom_to_feature(atom) for atom in mol.GetAtoms()]  # (n_atoms x n_atom_features)
    
    # 분자 단위 feature 추출
    global_features = global_feature_extractor(mol)  # (n_global_features,)
    
    # 하나로 합칠 수도 있음 (예: ML 모델 입력용)
    flattened_atom_features = np.array(atom_features).flatten()
    final_feature_vector = np.concatenate([flattened_atom_features, global_features])
    print(final_feature_vector)