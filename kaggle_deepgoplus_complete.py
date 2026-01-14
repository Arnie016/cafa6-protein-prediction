"""
🏆 CAFA-6 DeepGOPlus - COMPLETE KAGGLE NOTEBOOK
================================================
Runs entirely on Kaggle. Submits directly to competition.

REQUIRED DATASETS:
- cafa-6-protein-function-prediction (competition)
- zmcxjt/cafa5-train-test-data (PROTGOAT embeddings) 

RUN WITH: GPU T4 x2
TIME: ~4-6 hours
"""

# ============================================================================
# CELL 1: SETUP AND IMPORTS
# ============================================================================

!pip install biopython -q

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from Bio import SeqIO
from pathlib import Path
from datetime import datetime
import gc
import os

print("="*80)
print("🏆 DeepGOPlus for CAFA-6")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 2: CONFIGURATION
# ============================================================================

CONFIG = {
    'max_seq_len': 500,
    'cnn_filters': [32, 64, 128],
    'cnn_kernel_sizes': [8, 16, 32],
    'fusion_dim': 512,
    'hidden_dim': 256,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'epochs': 10,
    'batch_size': 128,
    'n_folds': 3,
    'n_terms': 500,
}

PATHS = {
    'protgoat': '/kaggle/input/cafa5-train-test-data/',
    'cafa6': '/kaggle/input/cafa-6-protein-function-prediction/',
    'output': '/kaggle/working/',
}

# Amino acid vocabulary
AA_VOCAB = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_VOCAB['X'] = 0

def encode_sequence(seq, max_len):
    encoded = [AA_VOCAB.get(aa, 0) for aa in seq[:max_len].upper()]
    encoded = encoded + [0] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)

# ============================================================================
# CELL 3: MODEL ARCHITECTURE
# ============================================================================

class SequenceCNN(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=32, filters=[32,64,128], kernels=[8,16,32]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, f, k, padding=k//2)
            for f, k in zip(filters, kernels)
        ])
        self.output_dim = sum(filters)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_outputs = [F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(-1) for conv in self.convs]
        return torch.cat(conv_outputs, dim=-1)

class FusionLayer(nn.Module):
    def __init__(self, cnn_dim, plm_dim, fusion_dim, dropout):
        super().__init__()
        proj_dim = fusion_dim // 3
        self.cnn_proj = nn.Linear(cnn_dim, proj_dim)
        self.plm_proj = nn.Linear(plm_dim, proj_dim)
        self.diamond_proj = nn.Linear(1, proj_dim)
        self.output = nn.Sequential(
            nn.Linear(proj_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, cnn_feat, plm_feat, diamond_score):
        h = torch.cat([
            F.relu(self.cnn_proj(cnn_feat)),
            F.relu(self.plm_proj(plm_feat)),
            F.relu(self.diamond_proj(diamond_score))
        ], dim=-1)
        return self.output(h)

class DeepGOPlus(nn.Module):
    def __init__(self, config, n_terms):
        super().__init__()
        self.seq_cnn = SequenceCNN(filters=config['cnn_filters'], kernels=config['cnn_kernel_sizes'])
        self.fusion = FusionLayer(self.seq_cnn.output_dim, 2560, config['fusion_dim'], config['dropout'])
        self.head = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], n_terms),
            nn.Sigmoid()
        )
        
    def forward(self, seq, plm, diamond):
        cnn_feat = self.seq_cnn(seq)
        fused = self.fusion(cnn_feat, plm, diamond)
        return self.head(fused)

class DeepGODataset(Dataset):
    def __init__(self, sequences, plm_embeddings, diamond_scores, labels=None):
        self.sequences = sequences
        self.plm_embeddings = torch.tensor(plm_embeddings, dtype=torch.float32)
        self.diamond_scores = torch.tensor(diamond_scores, dtype=torch.float32).unsqueeze(-1)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            return seq, self.plm_embeddings[idx], self.diamond_scores[idx], self.labels[idx]
        return seq, self.plm_embeddings[idx], self.diamond_scores[idx]

# ============================================================================
# CELL 4: LOAD DATA
# ============================================================================

print("\n[1/6] Loading PROTGOAT embeddings...")
protgoat_dir = Path(PATHS['protgoat'])

# Load ESM2 embeddings
esm2_test_emb = np.load(protgoat_dir / "ESM2_3B_test_embeddings_sorted.npy")
esm2_test_labels = np.load(protgoat_dir / "ESM2_3B_test_labels_sorted.npy", allow_pickle=True)
esm2_train_emb = np.load(protgoat_dir / "ESM2_3B_train_embeddings_sorted.npy")
esm2_train_labels = np.load(protgoat_dir / "ESM2_3B_train_labels_sorted.npy", allow_pickle=True)
print(f"  ESM2 test: {esm2_test_emb.shape}, train: {esm2_train_emb.shape}")

# Create lookup
protgoat_lookup = {}
for i, label in enumerate(esm2_test_labels):
    protgoat_lookup[str(label)] = ('test', i)
for i, label in enumerate(esm2_train_labels):
    if str(label) not in protgoat_lookup:
        protgoat_lookup[str(label)] = ('train', i)
print(f"  Total PROTGOAT proteins: {len(protgoat_lookup):,}")

# ============================================================================
# CELL 5: LOAD CAFA-6 DATA
# ============================================================================

print("\n[2/6] Loading CAFA-6 data...")
cafa6_dir = Path(PATHS['cafa6'])

def extract_accession(header):
    return header.split('|')[1] if '|' in header else header

# Load train
train_ids, train_seqs, train_acc = [], [], []
for record in SeqIO.parse(cafa6_dir / "Train/train_sequences.fasta", "fasta"):
    acc = extract_accession(record.id)
    train_ids.append(record.id)
    train_acc.append(acc)
    train_seqs.append(encode_sequence(str(record.seq), CONFIG['max_seq_len']))
print(f"  Train: {len(train_acc):,}")

# Load test
test_ids, test_seqs, test_acc = [], [], []
for record in SeqIO.parse(cafa6_dir / "Test/testsuperset.fasta", "fasta"):
    acc = extract_accession(record.id)
    test_ids.append(record.id)
    test_acc.append(acc)
    test_seqs.append(encode_sequence(str(record.seq), CONFIG['max_seq_len']))
print(f"  Test: {len(test_acc):,}")

X_train_seq = np.array(train_seqs)
X_test_seq = np.array(test_seqs)

# Extract PLM embeddings
print("\n[3/6] Extracting PLM embeddings...")
EMB_DIM = 2560
X_train_plm = np.zeros((len(train_acc), EMB_DIM), dtype=np.float32)
X_test_plm = np.zeros((len(test_acc), EMB_DIM), dtype=np.float32)

train_found, test_found = 0, 0
for i, acc in enumerate(train_acc):
    if acc in protgoat_lookup:
        src, idx = protgoat_lookup[acc]
        X_train_plm[i] = esm2_test_emb[idx] if src == 'test' else esm2_train_emb[idx]
        train_found += 1

for i, acc in enumerate(test_acc):
    if acc in protgoat_lookup:
        src, idx = protgoat_lookup[acc]
        X_test_plm[i] = esm2_test_emb[idx] if src == 'test' else esm2_train_emb[idx]
        test_found += 1

print(f"  Train PLM: {train_found}/{len(train_acc)} ({100*train_found/len(train_acc):.1f}%)")
print(f"  Test PLM: {test_found}/{len(test_acc)} ({100*test_found/len(test_acc):.1f}%)")

# Free memory
del esm2_test_emb, esm2_train_emb, protgoat_lookup
gc.collect()

# Placeholder DIAMOND scores
X_train_diamond = np.ones(len(train_acc), dtype=np.float32) * 0.5
X_test_diamond = np.ones(len(test_acc), dtype=np.float32) * 0.5

# ============================================================================
# CELL 6: LOAD LABELS
# ============================================================================

print("\n[4/6] Loading labels...")
df_terms = pd.read_csv(cafa6_dir / "Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
df_terms = df_terms[df_terms['EntryID'] != 'EntryID']
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
train_acc_to_idx = {acc: i for i, acc in enumerate(train_acc)}
print(f"  Proteins with labels: {len(protein_to_terms):,}")

# ============================================================================
# CELL 7: TRAINING
# ============================================================================

print("\n[5/6] Training DeepGOPlus...")
all_predictions = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*50}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*50}")
    
    aspect_df = df_terms[df_terms['aspect'] == aspect]
    terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
    n_terms = len(terms)
    
    y_train = np.zeros((len(train_acc), n_terms), dtype=np.float32)
    for acc, prot_terms in protein_to_terms.items():
        if acc in train_acc_to_idx:
            idx = train_acc_to_idx[acc]
            for j, term in enumerate(terms):
                if term in prot_terms:
                    y_train[idx, j] = 1
    
    print(f"  Terms: {n_terms}, Positive labels: {int(y_train.sum()):,}")
    
    aspect_preds = np.zeros((len(test_acc), n_terms), dtype=np.float32)
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train_plm, strat)):
        print(f"\n  Fold {fold+1}/{CONFIG['n_folds']}")
        
        model = DeepGOPlus(CONFIG, n_terms).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.BCELoss()
        
        train_dataset = DeepGODataset(X_train_seq[tr_idx], X_train_plm[tr_idx], X_train_diamond[tr_idx], y_train[tr_idx])
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        model.train()
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            for seq, plm, diamond, labels in train_loader:
                seq, plm, diamond, labels = seq.to(device), plm.to(device), diamond.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(seq, plm, diamond)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        model.eval()
        test_dataset = DeepGODataset(X_test_seq, X_test_plm, X_test_diamond)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
        
        fold_preds = []
        with torch.no_grad():
            for seq, plm, diamond in test_loader:
                seq, plm, diamond = seq.to(device), plm.to(device), diamond.to(device)
                outputs = model(seq, plm, diamond)
                fold_preds.append(outputs.cpu().numpy())
        
        aspect_preds += np.vstack(fold_preds)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    aspect_preds /= CONFIG['n_folds']
    
    for i, acc in enumerate(test_acc):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_predictions.append({'protein': acc, 'term': term, 'score': round(score, 6)})

# ============================================================================
# CELL 8: SAVE AND SUBMIT
# ============================================================================

print("\n[6/6] Creating submission...")
df_preds = pd.DataFrame(all_predictions)
print(f"  Total predictions: {len(df_preds):,}")
print(f"  Proteins: {df_preds['protein'].nunique():,}")

# Save
df_preds.to_csv('/kaggle/working/submission.tsv', sep='\t', index=False, header=False)
print("  Saved: /kaggle/working/submission.tsv")

# Check size
import os
size_mb = os.path.getsize('/kaggle/working/submission.tsv') / 1024 / 1024
print(f"  Size: {size_mb:.1f} MB")

print("\n" + "="*80)
print("🎉 DEEPGOPLUS COMPLETE!")
print("="*80)

# ============================================================================
# CELL 9: SUBMIT DIRECTLY TO COMPETITION
# ============================================================================

# Option 1: Submit via API (if you have credentials)
# Uncomment and set your credentials:

# import os
# os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
# os.environ['KAGGLE_KEY'] = 'YOUR_API_KEY'
# !kaggle competitions submit -c cafa-6-protein-function-prediction -f /kaggle/working/submission.tsv -m "DeepGOPlus"

# Option 2: Create downloadable link
from IPython.display import HTML, display
import base64

# If file is small enough, create download link
if size_mb < 100:
    with open('/kaggle/working/submission.tsv', 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    display(HTML(f'''
    <a download="submission.tsv" href="data:text/plain;base64,{data}" 
       style="font-size:24px; padding:15px 30px; background:#28a745; color:white; 
              border-radius:8px; text-decoration:none; display:inline-block; margin:20px 0;">
       📥 CLICK TO DOWNLOAD SUBMISSION
    </a>
    '''))
else:
    print("File too large for direct download. Use Save Version → Output tab")

print("\nTo submit manually:")
print("1. Click 'Save Version' (top right)")
print("2. Go to Output tab")
print("3. Click submission.tsv → Submit to Competition")
