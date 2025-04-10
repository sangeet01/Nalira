# metrics.py - Reward and validation metrics for Nalira pipeline

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from config import (
    ESOL_WEIGHT, XLOGP3_WEIGHT, SASCORE_WEIGHT, COMPLEXITY_WEIGHT,
    ESOL_TARGET, XLOGP3_MIN, XLOGP3_MAX, SASCORE_TARGET, TANIMOTO_THRESHOLD
)

# Solubility (ESOL) calculation
def calc_esol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Delaney's ESOL model: logS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP
    logP = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    esol = 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    return esol

# Lipophilicity (XLogP3) calculation
def calc_xlogp3(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)  # RDKit's approximation of XLogP3

# Synthetic Accessibility (SAscore)
def calc_sascore(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Simplified SAscore: based on fragment contributions and complexity
    num_atoms = mol.GetNumAtoms()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_chiral = len(Chem.FindMolChiralCenters(mol))
    sascore = 2.0 + 0.1 * num_atoms + 0.5 * num_rings + num_chiral  # Rough estimate
    return min(sascore, 10)  # Cap at 10 for practicality

# Complexity penalty
def calc_complexity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_chiral = len(Chem.FindMolChiralCenters(mol))
    penalty = 0
    if num_rings > 10:
        penalty += (num_rings - 10) * 0.5  # Penalize >10-membered rings
    if num_chiral > 5:
        penalty += (num_chiral - 5) * 0.5  # Penalize >5 chiral centers
    return penalty

# Combined reward for RL
def calc_reward(smiles):
    esol = calc_esol(smiles)
    xlogp3 = calc_xlogp3(smiles)
    sascore = calc_sascore(smiles)
    complexity = calc_complexity(smiles)
    if any(v is None for v in [esol, xlogp3, sascore]):
        return -10  # Harsh penalty for invalid SMILES
    reward = (
        ESOL_WEIGHT * max(0, esol - ESOL_TARGET) +  # Positive if > -3
        XLOGP3_WEIGHT * (1 if XLOGP3_MIN <= xlogp3 <= XLOGP3_MAX else 0) +  # 1 if in range
        SASCORE_WEIGHT * (SASCORE_TARGET - sascore) -  # Negative if > 4
        COMPLEXITY_WEIGHT * complexity
    )
    return reward

# Success check
def is_successful(smiles):
    esol = calc_esol(smiles)
    xlogp3 = calc_xlogp3(smiles)
    sascore = calc_sascore(smiles)
    if any(v is None for v in [esol, xlogp3, sascore]):
        return False
    return (
        esol > ESOL_TARGET and
        XLOGP3_MIN <= xlogp3 <= XLOGP3_MAX and
        sascore < SASCORE_TARGET
    )

# Tanimoto similarity for diversity
def calc_tanimoto(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 1.0  # Max similarity if invalid
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)

# Diversity check across analogs
def check_diversity(smiles_list):
    if len(smiles_list) < 2:
        return True
    for i in range(len(smiles_list)):
        for j in range(i + 1, len(smiles_list)):
            if calc_tanimoto(smiles_list[i], smiles_list[j]) > TANIMOTO_THRESHOLD:
                return False
    return True