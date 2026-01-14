#!/usr/bin/env python3
"""
PROTGOAT-LITE: Train Neural Network & Generate Predictions
===========================================================
Uses ESM2-3B embeddings to predict protein functions.
Expected F-max: 0.35-0.45
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
print("🧬 PROTGOAT-LITE: Neural Network Training")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
EMB_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
OUTPUT_DIR = Path("/Volumes/TRANSCEND/protgoat_lite/predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

# Config
CONFIG = {
    'hidden_dim': 512,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'epochs': 15,
    'batch_size': 256,
    'n_folds': 3,
    'n_terms_per_aspect': 500,
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Load embeddings
print("\n[1/5] Loading embeddings...")
X_train = np.load(EMB_DIR / "cafa6_train_esm2.npy")
X_test = np.load(EMB_DIR / "cafa6_test_esm2.npy")
train_ids = np.load(EMB_DIR / "train_ids.npy", allow_pickle=True)
test_ids = np.load(EMB_DIR / "test_ids.npy", allow_pickle=True)
print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# Load training labels
print("\n[2/5] Loading training labels...")
df_terms = pd.read_csv("Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
print(f"  Annotations: {len(df_terms):,}")

# Get top terms per aspect
top_terms = {}
for aspect in ['F', 'P', 'C']:
    aspect_df = df_terms[df_terms['aspect'] == aspect]
    top_terms[aspect] = aspect_df['term'].value_counts().head(CONFIG['n_terms_per_aspect']).index.tolist()
    print(f"  Aspect {aspect}: {len(top_terms[aspect])} terms")

# Neural network
class ProteinNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class ProteinDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Helper to get protein ID match
def match_protein_id(train_id, protein_to_terms):
    """Match train ID to protein_to_terms keys."""
    # Try exact match
    if train_id in protein_to_terms:
        return train_id
    # Try with sp| prefix
    for key in protein_to_terms.keys():
        if train_id in key:
            return key
    return None

# Training
print("\n[3/5] Training neural networks...")
all_predictions = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*50}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*50}")
    
    terms = top_terms[aspect]
    n_terms = len(terms)
    
    # Create labels
    y_train = np.zeros((len(train_ids), n_terms), dtype=np.float32)
    for i, tid in enumerate(train_ids):
        matched_key = match_protein_id(str(tid), protein_to_terms)
        if matched_key:
            protein_terms = protein_to_terms[matched_key]
            for j, term in enumerate(terms):
                if term in protein_terms:
                    y_train[i, j] = 1
    
    print(f"  Labels created: {y_train.sum():.0f} positive labels")
    
    # Train
    aspect_preds = np.zeros((len(test_ids), n_terms), dtype=np.float32)
    n_models = 0
    
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat_labels = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, strat_labels)):
        print(f"  Fold {fold+1}/{CONFIG['n_folds']}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = ProteinNN(2560, CONFIG['hidden_dim'], n_terms, CONFIG['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(ProteinDataset(X_tr, y_tr), batch_size=CONFIG['batch_size'], shuffle=True)
        
        model.train()
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Predict
        model.eval()
        test_loader = DataLoader(ProteinDataset(X_test), batch_size=CONFIG['batch_size'])
        
        fold_preds = []
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                fold_preds.append(outputs.cpu().numpy())
        
        aspect_preds += np.vstack(fold_preds)
        n_models += 1
        
        del model, optimizer
        gc.collect()
    
    aspect_preds /= n_models
    
    # Save predictions
    print(f"  Converting to submission format...")
    for i, tid in enumerate(test_ids):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_predictions.append({'protein': str(tid), 'term': term, 'score': score})

print(f"\n[4/5] Creating submission...")
df_preds = pd.DataFrame(all_predictions)
print(f"  Total predictions: {len(df_preds):,}")
print(f"  Proteins: {df_preds['protein'].nunique():,}")

# Save raw predictions
raw_file = OUTPUT_DIR / "protgoat_lite_raw.tsv"
df_preds.to_csv(raw_file, sep='\t', index=False, header=False)
print(f"  Saved: {raw_file}")

# Apply GO propagation
print("\n[5/5] Applying GO hierarchy propagation...")
import subprocess
final_file = OUTPUT_DIR / "protgoat_lite_submission.tsv"
subprocess.run([
    'python3', 'src/propagate_hierarchy.py',
    '--obo', 'Train/go-basic.obo',
    '--infile', str(raw_file),
    '--outfile', str(final_file),
    '--min_score', '0.01'
], check=True)

df_final = pd.read_csv(final_file, sep='\t', names=['protein', 'term', 'score'])
print(f"  Final predictions: {len(df_final):,}")

print("\n" + "="*80)
print("🎉 PROTGOAT-LITE COMPLETE!")
print("="*80)
print(f"Predictions: {len(df_final):,}")
print(f"File: {final_file}")
print(f"Expected F-max: 0.35-0.45")
print(f"Finished: {datetime.now()}")
