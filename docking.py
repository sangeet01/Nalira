# docking.py - Optional AutoDock Vina docking for Nalira pipeline

from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
from config import DOCKING_THRESHOLD

# Prepare ligand from SMILES
def prepare_ligand(smiles, output_file="ligand.pdbqt"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    # Write to PDBQT
    with open(output_file, "w") as f:
        f.write(Chem.MolToPDBBlock(mol))
    return output_file

# Prepare receptor from PDB ID (assumes pre-downloaded PDB)
def prepare_receptor(pdb_id, output_file="receptor.pdbqt"):
    pdb_file = f"{pdb_id}.pdb"
    try:
        # Basic prep: assumes PDB is cleaned (no waters, ligands)
        v = Vina()
        v.set_receptor(pdb_file)
        v.write_pdbqt(output_file)
        return output_file
    except Exception:
        return None

# Run docking
def run_docking(smiles, pdb_id, center=(0, 0, 0), box_size=(20, 20, 20)):
    ligand_file = prepare_ligand(smiles)
    if ligand_file is None:
        return None
    receptor_file = prepare_receptor(pdb_id)
    if receptor_file is None:
        return None
    
    # Initialize Vina
    v = Vina()
    v.set_receptor(receptor_file)
    v.set_ligand_from_file(ligand_file)
    
    # Define docking box (default center and size)
    v.compute_vina_maps(center=center, box_size=box_size)
    
    # Dock
    v.dock(exhaustiveness=8, n_poses=1)  # Single pose for speed
    energy = v.energies()[0][0]  # First pose energy (kcal/mol)
    
    # Clean up (optional in raw form)
    import os
    os.remove(ligand_file)
    os.remove(receptor_file)
    
    return energy

# Check docking success
def is_docking_successful(smiles, pdb_id):
    energy = run_docking(smiles, pdb_id)
    if energy is None:
        return False
    return energy < DOCKING_THRESHOLD  # ΔG < -7 kcal/mol

# Main function for integration
def dock_smiles(smiles, pdb_id):
    energy = run_docking(smiles, pdb_id)
    if energy is None:
        return {"energy": None, "success": False}
    success = energy < DOCKING_THRESHOLD
    return {"energy": energy, "success": success}