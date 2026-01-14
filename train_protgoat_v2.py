#!/usr/bin/env python3
"""
PROTGOAT-LITE v2: Fixed protein ID matching
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧬 PROTGOAT-LITE v2: Neural Network Training")
print("="*80)
print(f"Started: {datetime.now()}")

EMB_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
OUTPUT_DIR = EMB_DIR / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {'hidden_dim': 512, 'dropout': 0.3, 'lr': 1e-3, 'epochs': 15, 'batch_size': 256, 'n_folds': 3, 'n_terms': 500}

device = torch.device("cpu")  # Force CPU - MPS is unstable
print(f"Device: {device}")

# Load embeddings
print("\n[1/5] Loading data...")
X_train = np.load(EMB_DIR / "cafa6_train_esm2.npy")
X_test = np.load(EMB_DIR / "cafa6_test_esm2.npy")
train_acc = np.load(EMB_DIR / "train_accessions.npy", allow_pickle=True)
test_acc = np.load(EMB_DIR / "test_accessions.npy", allow_pickle=True)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# Load labels
df_terms = pd.read_csv("Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
df_terms = df_terms[df_terms['EntryID'] != 'EntryID']  # Skip header if exists
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
print(f"  Proteins with labels: {len(protein_to_terms):,}")

# Map train accessions to indices
train_acc_to_idx = {acc: i for i, acc in enumerate(train_acc)}

class ProteinNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ProteinDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

print("\n[2/5] Training...")
all_preds = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*50}\nASPECT: {aspect}\n{'='*50}")
    
    # Get top terms
    aspect_df = df_terms[df_terms['aspect'] == aspect]
    terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
    
    # Create labels
    y_train = np.zeros((len(train_acc), len(terms)), dtype=np.float32)
    matched = 0
    for acc, protein_terms in protein_to_terms.items():
        if acc in train_acc_to_idx:
            idx = train_acc_to_idx[acc]
            matched += 1
            for j, term in enumerate(terms):
                if term in protein_terms:
                    y_train[idx, j] = 1
    
    pos_labels = int(y_train.sum())
    print(f"  Matched: {matched:,}, Positive labels: {pos_labels:,}")
    
    if pos_labels == 0:
        print("  SKIPPING - no labels!")
        continue
    
    # Train
    aspect_preds = np.zeros((len(test_acc), len(terms)), dtype=np.float32)
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, strat)):
        print(f"  Fold {fold+1}/{CONFIG['n_folds']}")
        
        model = ProteinNN(2560, CONFIG['hidden_dim'], len(terms), CONFIG['dropout']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        crit = nn.BCELoss()
        
        loader = DataLoader(ProteinDataset(X_train[tr_idx], y_train[tr_idx]), batch_size=CONFIG['batch_size'], shuffle=True)
        
        model.train()
        for epoch in range(CONFIG['epochs']):
            loss_sum = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = model(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss={loss_sum/len(loader):.4f}")
        
        # Predict
        model.eval()
        test_loader = DataLoader(ProteinDataset(X_test), batch_size=CONFIG['batch_size'])
        preds = []
        with torch.no_grad():
            for bx in test_loader:
                bx = bx.to(device)
                preds.append(model(bx).cpu().numpy())
        aspect_preds += np.vstack(preds)
        del model, opt
        gc.collect()
    
    aspect_preds /= CONFIG['n_folds']
    
    # Convert to submission
    for i, acc in enumerate(test_acc):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_preds.append({'protein': acc, 'term': term, 'score': round(score, 6)})

print(f"\n[3/5] Saving...")
df_preds = pd.DataFrame(all_preds)
print(f"  Total: {len(df_preds):,} predictions")

raw_file = OUTPUT_DIR / "protgoat_raw.tsv"
df_preds.to_csv(raw_file, sep='\t', index=False, header=False)

print("\n[4/5] GO propagation...")
import subprocess
final_file = OUTPUT_DIR / "protgoat_submission.tsv"
subprocess.run(['python3', 'src/propagate_hierarchy.py', '--obo', 'Train/go-basic.obo',
                '--infile', str(raw_file), '--outfile', str(final_file), '--min_score', '0.01'], check=True)

df_final = pd.read_csv(final_file, sep='\t', names=['protein', 'term', 'score'])
print(f"  Final: {len(df_final):,}")

print(f"\n[5/5] Copying to project...")
import shutil
shutil.copy(final_file, "/Users/hema/Desktop/cafa-6-protein-function-prediction/protgoat_submission.tsv")

print("\n" + "="*80)
print("🎉 DONE!")
print("="*80)
print(f"File: protgoat_submission.tsv")
print(f"Predictions: {len(df_final):,}")
print(f"Finished: {datetime.now()}")
