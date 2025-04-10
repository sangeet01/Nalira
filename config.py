# config.py - Hyperparameters and settings for Nalira pipeline

# Pre-training settings (PubChem, Section 3.3)
EPOCHS_PRETRAIN = 20
LEARNING_RATE_PRETRAIN = 0.001
BATCH_SIZE_PRETRAIN = 128

# Fine-tuning settings (COCONUT, Section 3.4)
EPOCHS_FINETUNE = 10
LEARNING_RATE_FINETUNE = 0.0001
BATCH_SIZE_FINETUNE = 128

# Reinforcement Learning settings (Section 3.4)
RL_ITERATIONS = 1000
ESOL_WEIGHT = 0.4          # Reward weight for solubility (logS > -3)
XLOGP3_WEIGHT = 0.3        # Reward weight for lipophilicity (1–3)
SASCORE_WEIGHT = -0.3      # Penalty weight for synthetic accessibility (< 4)
COMPLEXITY_WEIGHT = -0.1   # Penalty for >10-membered rings or >5 chiral centers
ESOL_TARGET = -3           # logS > -3
XLOGP3_MIN = 1             # XLogP3 range 1–3
XLOGP3_MAX = 3
SASCORE_TARGET = 4         # SAscore < 4

# Input settings (Section 3.2)
DEFAULT_ANALOGS = 100      # Default number of analogs per SMILES
RUNTIME_OPTIONS = ["Local CPU", "Colab GPU/CPU"]

# Target prediction settings (Section 3.5)
DTI_TOP_N = 3              # Predict top 3 receptors

# Validation settings (Section 3.5)
TANIMOTO_THRESHOLD = 0.4   # Diversity: Tanimoto < 0.4
DOCKING_THRESHOLD = -7     # Optional docking: ΔG < -7 kcal/mol

# File paths (Section 2)
PUBCHEM_SMILES_FILE = "pubchem_smiles.csv"
COCONUT_SMILES_FILE = "coconut_smiles.csv"
NPASS_BIOACTIVITY_FILE = "npass_bioactivity.csv"
OUTPUT_FILE = "nalira_output.csv"