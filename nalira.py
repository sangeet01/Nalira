# nalira.py - Main pipeline for Nalira: NP optimization and receptor prediction

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from rdkit import Chem
from data_utils import preprocess_input, get_user_input
from metrics import calc_reward, is_successful, check_diversity
from docking import dock_smiles
from config import (
    EPOCHS_PRETRAIN, LEARNING_RATE_PRETRAIN, BATCH_SIZE_PRETRAIN,
    EPOCHS_FINETUNE, LEARNING_RATE_FINETUNE, BATCH_SIZE_FINETUNE,
    RL_ITERATIONS, DEFAULT_ANALOGS, DTI_TOP_N, PUBCHEM_SMILES_FILE,
    COCONUT_SMILES_FILE, NPASS_BIOACTIVITY_FILE, OUTPUT_FILE
)

# Simple SMILES dataset
class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]

# RNN model (REINVENT-inspired)
class ReinventRNN(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# Simplified SMILES tokenizer (raw, no pre-built vocab)
def tokenize_smiles(smiles):
    return [ord(c) % 128 for c in smiles]  # ASCII-based, cap at 128

def detokenize_smiles(tokens):
    return "".join(chr(t) for t in tokens if t < 128)

# Pre-training
def pretrain_model(model, smiles_file, epochs, lr, batch_size):
    smiles_list = preprocess_input(smiles_file, "csv")[0]  # First batch
    dataset = SmilesDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in loader:
            tokens = [tokenize_smiles(s) for s in batch]
            max_len = max(len(t) for t in tokens)
            x = torch.tensor([t + [0] * (max_len - len(t)) for t in tokens], dtype=torch.long)
            y = x[:, 1:]  # Shift for next-token prediction
            x = x[:, :-1]
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.view(-1, 128), y.reshape(-1))
            loss.backward()
            optimizer.step()
    return model

# Fine-tuning
def finetune_model(model, smiles_file, epochs, lr, batch_size):
    return pretrain_model(model, smiles_file, epochs, lr, batch_size)  # Same logic, different data

# RL optimization
def optimize_with_rl(model, input_smiles, iterations, num_analogs):
    analogs = []
    for _ in range(iterations):
        x = torch.tensor([tokenize_smiles(input_smiles)], dtype=torch.long)
        out, hidden = model(x)
        probs = torch.softmax(out[0], dim=-1)
        for _ in range(num_analogs):
            sampled = []
            h = hidden
            for t in range(len(input_smiles)):  # Rough length match
                token = torch.multinomial(probs[t], 1).item()
                sampled.append(token)
                next_in = torch.tensor([[token]], dtype=torch.long)
                probs, h = model(next_in, h)
            new_smiles = detokenize_smiles(sampled)
            if Chem.MolFromSmiles(new_smiles):  # Valid SMILES only
                reward = calc_reward(new_smiles)
                if reward > 0:
                    analogs.append(new_smiles)
        # Simplified policy update (raw, no gradient clipping)
        model.train()
        loss = -sum(calc_reward(s) for s in analogs) / max(1, len(analogs))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return list(set(analogs))[:num_analogs]  # Deduplicate, cap at num_analogs

# Dummy DTI prediction (raw placeholder, no pre-trained model)
def predict_receptors(smiles, top_n):
    # Placeholder: real Deep DTI needs NPASS training
    return [(f"Target_{i}", 0.9 - i * 0.1) for i in range(top_n)]

# Main pipeline
def main():
    # Get user input
    print("Starting Nalira...")
    input_data = get_user_input()
    pdb_id = input("Include docking? Enter PDB ID (e.g., 1M17) or 'skip': ")
    
    # Initialize model
    model = ReinventRNN(vocab_size=128, hidden_size=512)
    
    # Pre-train
    print("Pre-training on PubChem...")
    model = pretrain_model(model, PUBCHEM_SMILES_FILE, EPOCHS_PRETRAIN,
                          LEARNING_RATE_PRETRAIN, BATCH_SIZE_PRETRAIN)
    
    # Fine-tune
    print("Fine-tuning on COCONUT...")
    model = finetune_model(model, COCONUT_SMILES_FILE, EPOCHS_FINETUNE,
                          LEARNING_RATE_FINETUNE, BATCH_SIZE_FINETUNE)
    
    # Process inputs
    results = []
    for batch in (input_data if isinstance(input_data[0], list) else [input_data]):
        for smiles in batch:
            # Optimize
            analogs = optimize_with_rl(model, smiles, RL_ITERATIONS, DEFAULT_ANALOGS)
            if not check_diversity(analogs):
                print(f"Warning: Diversity < {TANIMOTO_THRESHOLD} for {smiles}")
            
            # Predict receptors
            for analog in analogs:
                receptors = predict_receptors(analog, DTI_TOP_N)
                success = is_successful(analog)
                docking = dock_smiles(analog, pdb_id) if pdb_id != "skip" else {"energy": None, "success": False}
                results.append({
                    "smiles": analog,
                    "logS": calc_esol(analog),
                    "XLogP3": calc_xlogp3(analog),
                    "SAscore": calc_sascore(analog),
                    "success": success,
                    "diversity": check_diversity(analogs),
                    "receptor_predicted": receptors[0][0],
                    "score": receptors[0][1],
                    "docking_energy": docking["energy"]
                })
    
    # Save output
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()