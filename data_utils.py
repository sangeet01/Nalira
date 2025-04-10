# data_utils.py - Input preprocessing for Nalira pipeline

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from config import BATCH_SIZE_FINETUNE

# Validate and standardize SMILES
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Standardize with OpenBabel
    ob_conv = ob.OBConversion()
    ob_conv.SetInAndOutFormats("smi", "can")
    ob_mol = ob.OBMol()
    ob_conv.ReadString(ob_mol, smiles)
    canonical_smiles = ob_conv.WriteString(ob_mol).strip()
    return canonical_smiles

# Fetch SMILES from compound name via PubChem API
def fetch_smiles_from_name(name):
    try:
        compounds = pcp.get_compounds(name, "name")
        if not compounds:
            return None
        smiles = compounds[0].isomeric_smiles  # Use isomeric SMILES
        return validate_smiles(smiles)
    except Exception:
        return None

# Process .csv batch
def process_csv(file_path):
    df = pd.read_csv(file_path)
    if "smiles" not in df.columns:
        raise ValueError("CSV must contain 'smiles' column")
    # Validate and standardize each SMILES
    df["smiles"] = df["smiles"].apply(validate_smiles)
    # Remove invalid SMILES and duplicates
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
    return df["smiles"].tolist()

# Main preprocessing function
def preprocess_input(user_input, input_type="smiles"):
    if input_type == "smiles":
        smiles = validate_smiles(user_input)
        if smiles:
            return [smiles]
        raise ValueError("Invalid SMILES string")
    elif input_type == "name":
        smiles = fetch_smiles_from_name(user_input)
        if smiles:
            return [smiles]
        raise ValueError(f"No SMILES found for compound name: {user_input}")
    elif input_type == "csv":
        smiles_list = process_csv(user_input)
        if not smiles_list:
            raise ValueError("No valid SMILES in CSV")
        # Batch for efficiency
        return [
            smiles_list[i:i + BATCH_SIZE_FINETUNE]
            for i in range(0, len(smiles_list), BATCH_SIZE_FINETUNE)
        ]
    else:
        raise ValueError("Input type must be 'smiles', 'name', or 'csv'")

# User prompt handler (simplified for raw coding)
def get_user_input():
    input_str = input("Enter SMILES (e.g., CCO), name (e.g., artemisinin), or .csv path: ")
    if input_str.endswith(".csv"):
        return preprocess_input(input_str, "csv")
    elif len(input_str) < 20 and not any(c in input_str for c in "()[],"):  # Rough heuristic for names
        return preprocess_input(input_str, "name")
    else:
        return preprocess_input(input_str, "smiles")