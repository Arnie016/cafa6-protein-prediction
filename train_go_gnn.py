#!/usr/bin/env python3
"""
GO-Graph Neural Network for Protein Function Prediction
=========================================================
Uses message passing on GO-DAG to learn term relationships.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧬 GO-GRAPH NEURAL NETWORK")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
EMB_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
GRAPH_DIR = EMB_DIR / "go_graph"
OUTPUT_DIR = EMB_DIR / "go_gnn_predictions"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'input_dim': 2560,
    'hidden_dim': 256,
    'gnn_layers': 3,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'epochs': 20,
    'batch_size': 256,
    'n_folds': 3,
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# GRAPH NEURAL NETWORK LAYERS
# ============================================================================

class GraphConvLayer(nn.Module):
    """Graph Convolution Layer - message passing on GO-DAG."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # Message passing: aggregate neighbor features
        # adj is normalized adjacency matrix
        h = torch.matmul(adj, x)  # Aggregate neighbors
        h = self.linear(h)  # Transform
        return F.relu(h)

class GOGraphNN(nn.Module):
    """
    GO-Graph Neural Network
    1. Project protein embedding to per-GO-term features
    2. Message passing on GO-DAG
    3. Final prediction
    """
    def __init__(self, input_dim, hidden_dim, n_terms, n_layers, adj, dropout):
        super().__init__()
        
        # Register adjacency as buffer (moved to device with model)
        self.register_buffer('adj', adj)
        
        # Initial projection: protein embedding -> per-term features
        self.protein_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Per-term initial embedding
        self.term_embed = nn.Embedding(n_terms, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Final prediction
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.n_terms = n_terms
        
    def forward(self, protein_emb):
        batch_size = protein_emb.shape[0]
        
        # Project protein to hidden dim
        protein_h = self.protein_proj(protein_emb)  # (batch, hidden)
        
        # Initialize GO term node features
        term_indices = torch.arange(self.n_terms, device=protein_emb.device)
        term_h = self.term_embed(term_indices)  # (n_terms, hidden)
        
        # Message passing on GO-DAG
        for gnn in self.gnn_layers:
            term_h = gnn(term_h, self.adj)
        
        # Combine protein features with term features
        # Broadcast protein to all terms
        protein_h_expanded = protein_h.unsqueeze(1).expand(-1, self.n_terms, -1)  # (batch, n_terms, hidden)
        term_h_expanded = term_h.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_terms, hidden)
        
        # Concatenate
        combined = torch.cat([protein_h_expanded, term_h_expanded], dim=-1)  # (batch, n_terms, hidden*2)
        
        # Predict
        scores = self.output(combined).squeeze(-1)  # (batch, n_terms)
        
        return scores

class ProteinDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

# ============================================================================
# TRAINING
# ============================================================================

print("\n[1/4] Loading data...")
X_train = np.load(EMB_DIR / "cafa6_train_esm2.npy")
X_test = np.load(EMB_DIR / "cafa6_test_esm2.npy")
train_acc = np.load(EMB_DIR / "train_accessions.npy", allow_pickle=True)
test_acc = np.load(EMB_DIR / "test_accessions.npy", allow_pickle=True)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# Load labels
df_terms = pd.read_csv("Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
df_terms = df_terms[df_terms['EntryID'] != 'EntryID']
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
train_acc_to_idx = {acc: i for i, acc in enumerate(train_acc)}

print("\n[2/4] Training GO-GNN for each aspect...")

from sklearn.model_selection import StratifiedKFold
all_predictions = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*50}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*50}")
    
    # Load GO graph
    adj = np.load(GRAPH_DIR / f"adj_{aspect}.npy")
    terms = np.load(GRAPH_DIR / f"terms_{aspect}.npy", allow_pickle=True)
    n_terms = len(terms)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    
    print(f"  Terms: {n_terms}, Graph edges: {(adj > 0).sum()}")
    
    # Create labels
    y_train = np.zeros((len(train_acc), n_terms), dtype=np.float32)
    for acc, prot_terms in protein_to_terms.items():
        if acc in train_acc_to_idx:
            idx = train_acc_to_idx[acc]
            for j, term in enumerate(terms):
                if term in prot_terms:
                    y_train[idx, j] = 1
    
    pos_labels = int(y_train.sum())
    print(f"  Positive labels: {pos_labels:,}")
    
    # Train with cross-validation
    aspect_preds = np.zeros((len(test_acc), n_terms), dtype=np.float32)
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, strat)):
        print(f"\n  Fold {fold+1}/{CONFIG['n_folds']}")
        
        model = GOGraphNN(
            input_dim=CONFIG['input_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            n_terms=n_terms,
            n_layers=CONFIG['gnn_layers'],
            adj=adj_tensor,
            dropout=CONFIG['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(
            ProteinDataset(X_train[tr_idx], y_train[tr_idx]),
            batch_size=CONFIG['batch_size'], shuffle=True
        )
        
        model.train()
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Predict test
        model.eval()
        test_loader = DataLoader(ProteinDataset(X_test), batch_size=CONFIG['batch_size'])
        
        fold_preds = []
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                fold_preds.append(outputs.cpu().numpy())
        
        aspect_preds += np.vstack(fold_preds)
        
        del model, optimizer
        gc.collect()
    
    aspect_preds /= CONFIG['n_folds']
    
    # Convert to predictions
    print(f"  Converting predictions...")
    for i, acc in enumerate(test_acc):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_predictions.append({'protein': acc, 'term': term, 'score': round(score, 6)})

print(f"\n[3/4] Saving predictions...")
df_preds = pd.DataFrame(all_predictions)
print(f"  Total: {len(df_preds):,}")

raw_file = OUTPUT_DIR / "go_gnn_raw.tsv"
df_preds.to_csv(raw_file, sep='\t', index=False, header=False)

print("\n[4/4] GO propagation...")
import subprocess
final_file = OUTPUT_DIR / "go_gnn_submission.tsv"
subprocess.run(['python3', 'src/propagate_hierarchy.py', '--obo', 'Train/go-basic.obo',
                '--infile', str(raw_file), '--outfile', str(final_file), '--min_score', '0.01'], check=True)

df_final = pd.read_csv(final_file, sep='\t', names=['protein', 'term', 'score'])
print(f"  Final: {len(df_final):,}")

print("\n" + "="*80)
print("🎉 GO-GNN COMPLETE!")
print("="*80)
print(f"File: {final_file}")
print(f"Predictions: {len(df_final):,}")
print(f"Finished: {datetime.now()}")
