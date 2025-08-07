# chemical feature extractor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, MACCSkeys, AllChem, Crippen
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# basic library
import numpy as np
import pandas as pd

# SMARTS 패턴 정의 (논문 기반)
SMARTS_PATTERNS = {
    "has_imidazole": Chem.MolFromSmarts("n1cncc1"),
    "has_tertiary_amine": Chem.MolFromSmarts("[NX3]([C])[C]"),
    "has_furan": Chem.MolFromSmarts("c1ccoc1"),
    "has_acetylene": Chem.MolFromSmarts("C#C"), 
    "has_pyridine": Chem.MolFromSmarts("n1ccccc1"),
    "has_thiophene": Chem.MolFromSmarts("c1ccsc1")
}

def feature_extractor(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # 기본 물성 특징 15개
    physical_features = [
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

    # Morgan fingerprint (2048bit)
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    morgan_fp = generator.GetFingerprint(mol)
    morgan_features = list(morgan_fp)

    # MACCS fingerprint (167bit)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_features = [int(bit) for bit in maccs_fp.ToBitString()]

    # SMARTS 구조 플래그 (6개)
    smarts_flags = [int(mol.HasSubstructMatch(pat)) for pat in SMARTS_PATTERNS.values()]

    # 전체 특징 벡터
    all_features = physical_features + morgan_features + maccs_features + smarts_flags
    
    return all_features


# debugging
if __name__ == "__main__":
    print(len(feature_extractor("CCC")))