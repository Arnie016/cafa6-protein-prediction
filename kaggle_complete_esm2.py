"""
🧬 CAFA-6: Generate Missing Embeddings + Train
===============================================
Generates ESM2 embeddings for proteins not in PROTGOAT dataset.
Then trains and creates final submission.

REQUIRED DATASETS:
- cafa-6-protein-function-prediction (competition)

RUN WITH: GPU T4 x2
TIME: ~8-10 hours
"""

# ============================================================================
# CELL 1: SETUP
# ============================================================================

!pip install transformers biopython -q

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from pathlib import Path
from datetime import datetime
import gc
import os

print("="*80)
print("🧬 CAFA-6: Complete ESM2 Pipeline")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CELL 2: LOAD CAFA-6 DATA
# ============================================================================

print("\n[1/5] Loading CAFA-6 data...")

# Paths
CAFA6_DIR = Path("/kaggle/input/cafa-6-protein-function-prediction/")

# Load train
train_proteins = {}
for record in SeqIO.parse(CAFA6_DIR / "Train/train_sequences.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    train_proteins[acc] = str(record.seq)
print(f"  Train proteins: {len(train_proteins):,}")

# Load test  
test_proteins = {}
for record in SeqIO.parse(CAFA6_DIR / "Test/testsuperset.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    test_proteins[acc] = str(record.seq)
print(f"  Test proteins: {len(test_proteins):,}")

# Load labels
df_terms = pd.read_csv(CAFA6_DIR / "Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
df_terms = df_terms[df_terms['EntryID'] != 'EntryID']
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
print(f"  Train proteins with labels: {len(protein_to_terms):,}")

# ============================================================================
# CELL 3: GENERATE ESM2 EMBEDDINGS
# ============================================================================

print("\n[2/5] Loading ESM2 model...")

# Use ESM2-650M (fits on T4)
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name).to(device).eval()
print(f"  Model: {model_name}")

def get_esm2_embedding(sequences, batch_size=8, max_len=1022):
    """Generate ESM2 embeddings for a list of sequences."""
    embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        # Truncate long sequences
        batch = [seq[:max_len] for seq in batch]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len+2)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = esm_model(**inputs)
            # Mean pooling over sequence length
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())
        
        if (i // batch_size) % 100 == 0:
            print(f"    Processed {i + len(batch)}/{len(sequences)} sequences...")
        
        # Clear cache
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    
    return np.vstack(embeddings)

# Generate for train
print("\n  Generating train embeddings...")
train_accs = list(train_proteins.keys())
train_seqs = [train_proteins[acc] for acc in train_accs]
X_train = get_esm2_embedding(train_seqs, batch_size=4)
print(f"  Train embeddings: {X_train.shape}")

# Generate for test
print("\n  Generating test embeddings...")
test_accs = list(test_proteins.keys())
test_seqs = [test_proteins[acc] for acc in test_accs]
X_test = get_esm2_embedding(test_seqs, batch_size=4)
print(f"  Test embeddings: {X_test.shape}")

# Free model memory
del esm_model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# ============================================================================
# CELL 4: TRAIN NEURAL NETWORK
# ============================================================================

print("\n[3/5] Training neural network...")

CONFIG = {
    'hidden_dim': 512,
    'dropout': 0.3,
    'lr': 1e-3,
    'epochs': 15,
    'batch_size': 256,
    'n_folds': 3,
    'n_terms': 500,
}

EMB_DIM = X_train.shape[1]

class ProteinNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim),
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
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

# Create train label mapping
train_acc_to_idx = {acc: i for i, acc in enumerate(train_accs)}

from sklearn.model_selection import StratifiedKFold

all_predictions = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*50}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*50}")
    
    # Get top terms
    aspect_df = df_terms[df_terms['aspect'] == aspect]
    terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
    n_terms = len(terms)
    
    # Create labels
    y_train = np.zeros((len(train_accs), n_terms), dtype=np.float32)
    for acc, prot_terms in protein_to_terms.items():
        if acc in train_acc_to_idx:
            idx = train_acc_to_idx[acc]
            for j, term in enumerate(terms):
                if term in prot_terms:
                    y_train[idx, j] = 1
    
    print(f"  Terms: {n_terms}, Positive labels: {int(y_train.sum()):,}")
    
    # Train with CV
    aspect_preds = np.zeros((len(test_accs), n_terms), dtype=np.float32)
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, strat)):
        print(f"  Fold {fold+1}/{CONFIG['n_folds']}")
        
        model = ProteinNN(EMB_DIM, CONFIG['hidden_dim'], n_terms, CONFIG['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(
            ProteinDataset(X_train[tr_idx], y_train[tr_idx]),
            batch_size=CONFIG['batch_size'], shuffle=True
        )
        
        model.train()
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
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
            for bx in test_loader:
                bx = bx.to(device)
                fold_preds.append(model(bx).cpu().numpy())
        aspect_preds += np.vstack(fold_preds)
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    aspect_preds /= CONFIG['n_folds']
    
    # Convert to predictions
    for i, acc in enumerate(test_accs):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_predictions.append({'protein': acc, 'term': term, 'score': round(score, 6)})

# ============================================================================
# CELL 5: CREATE SUBMISSION
# ============================================================================

print("\n[4/5] Creating submission...")
df_preds = pd.DataFrame(all_predictions)
print(f"  Total predictions: {len(df_preds):,}")
print(f"  Proteins: {df_preds['protein'].nunique():,}")

# Save
df_preds.to_csv('/kaggle/working/submission.tsv', sep='\t', index=False, header=False)
print("  Saved: /kaggle/working/submission.tsv")

# Stats
import os
size_mb = os.path.getsize('/kaggle/working/submission.tsv') / 1024 / 1024
print(f"  Size: {size_mb:.1f} MB")

print("\n[5/5] Done!")
print("="*80)
print("🎉 SUBMISSION READY!")
print("="*80)
print(f"File: /kaggle/working/submission.tsv")
print(f"Predictions: {len(df_preds):,}")
print(f"Expected F-max: 0.30-0.40")
print("\nTo submit: Save Version → Output → Submit to Competition")
