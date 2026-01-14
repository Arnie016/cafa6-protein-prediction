"""
🏆 CAFA-6 WINNER-STYLE Model - DUAL GPU VERSION
================================================
Uses both T4 GPUs for 2x faster embedding generation!

RUN WITH: GPU T4 x2
TIME: ~5-6 hours (2x faster!)
"""

!pip install transformers biopython obonet -q

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import obonet
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import gc
import os

print("="*80)
print("🏆 CAFA-6 WINNER-STYLE Model - DUAL GPU")
print("="*80)

# Check GPUs
n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0")

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'hidden_dims': [1024, 512, 256, 128],
    'dropout': 0.4,
    'residual': True,
    'hierarchy_weight': 0.1,
    'lr': 5e-4,
    'epochs': 20,
    'batch_size': 128,
    'n_folds': 5,
    'n_terms': 1000,
    'esm_model': "facebook/esm2_t33_650M_UR50D",
    'max_seq_len': 1022,
}

# ============================================================================
# GO HIERARCHY
# ============================================================================

print("\n[1/6] Loading GO hierarchy...")
CAFA6_DIR = Path("/kaggle/input/cafa-6-protein-function-prediction/")

go_graph = obonet.read_obo(str(CAFA6_DIR / "Train/go-basic.obo"))
print(f"  GO terms: {len(go_graph):,}")

def get_ancestors(term, graph):
    ancestors = set()
    if term in graph:
        for parent in graph.predecessors(term):
            ancestors.add(parent)
            ancestors.update(get_ancestors(parent, graph))
    return ancestors

# ============================================================================
# DUAL-GPU EMBEDDING GENERATION
# ============================================================================

print("\n[2/6] Loading CAFA-6 data...")

train_proteins = {}
for record in SeqIO.parse(CAFA6_DIR / "Train/train_sequences.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    train_proteins[acc] = str(record.seq)

test_proteins = {}
for record in SeqIO.parse(CAFA6_DIR / "Test/testsuperset.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    test_proteins[acc] = str(record.seq)

print(f"  Train: {len(train_proteins):,}, Test: {len(test_proteins):,}")

df_terms = pd.read_csv(CAFA6_DIR / "Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
df_terms = df_terms[df_terms['EntryID'] != 'EntryID']
protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()

print("\n[3/6] Generating ESM2 embeddings (DUAL GPU)...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['esm_model'])

# Load model on both GPUs
if n_gpus >= 2:
    print("  Using DUAL GPU mode!")
    esm_model_0 = AutoModel.from_pretrained(CONFIG['esm_model']).to("cuda:0").eval()
    esm_model_1 = AutoModel.from_pretrained(CONFIG['esm_model']).to("cuda:1").eval()
else:
    print("  Single GPU mode")
    esm_model_0 = AutoModel.from_pretrained(CONFIG['esm_model']).to("cuda:0").eval()
    esm_model_1 = None

def get_embeddings_dual_gpu(proteins, batch_size=8):
    """Generate embeddings using both GPUs in parallel."""
    accs = list(proteins.keys())
    seqs = [proteins[acc][:CONFIG['max_seq_len']] for acc in accs]
    
    # Split into two halves
    mid = len(seqs) // 2
    seqs_0, seqs_1 = seqs[:mid], seqs[mid:]
    accs_0, accs_1 = accs[:mid], accs[mid:]
    
    def process_on_gpu(sequences, model, gpu_id):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG['max_seq_len']+2)
            inputs = {k: v.to(f"cuda:{gpu_id}") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(emb.cpu().numpy())
            
            if (i // batch_size) % 100 == 0:
                print(f"    GPU {gpu_id}: {i+len(batch)}/{len(sequences)}")
                torch.cuda.empty_cache()
        
        return np.vstack(embeddings)
    
    if esm_model_1 is not None:
        # Process in parallel using threads
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(process_on_gpu, seqs_0, esm_model_0, 0)
            future_1 = executor.submit(process_on_gpu, seqs_1, esm_model_1, 1)
            emb_0 = future_0.result()
            emb_1 = future_1.result()
        
        return accs_0 + accs_1, np.vstack([emb_0, emb_1])
    else:
        emb = process_on_gpu(seqs, esm_model_0, 0)
        return accs, emb

print("  Train embeddings...")
train_accs, X_train = get_embeddings_dual_gpu(train_proteins, batch_size=4)
print(f"    Shape: {X_train.shape}")

print("  Test embeddings...")
test_accs, X_test = get_embeddings_dual_gpu(test_proteins, batch_size=4)
print(f"    Shape: {X_test.shape}")

del esm_model_0, esm_model_1, tokenizer
gc.collect()
torch.cuda.empty_cache()

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.act(x + self.net(x)))

class WinnerStyleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_terms, dropout, use_residual=True):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
        self.backbone = nn.Sequential(*layers)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_terms),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)
        return self.head(x)

class HierarchicalLoss(nn.Module):
    def __init__(self, parent_child_pairs, weight=0.1):
        super().__init__()
        self.parent_child_pairs = parent_child_pairs
        self.weight = weight
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        if len(self.parent_child_pairs) > 0:
            hierarchy_loss = 0
            for parent_idx, child_idx in self.parent_child_pairs:
                violation = F.relu(pred[:, child_idx] - pred[:, parent_idx])
                hierarchy_loss += violation.mean()
            hierarchy_loss /= len(self.parent_child_pairs)
            return bce_loss + self.weight * hierarchy_loss
        
        return bce_loss

# ============================================================================
# TRAINING (uses DataParallel for both GPUs)
# ============================================================================

print("\n[4/6] Training winner-style model...")

train_acc_to_idx = {acc: i for i, acc in enumerate(train_accs)}
all_predictions = []

for aspect in ['F', 'P', 'C']:
    print(f"\n{'='*60}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*60}")
    
    aspect_df = df_terms[df_terms['aspect'] == aspect]
    terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
    term_to_idx = {t: i for i, t in enumerate(terms)}
    n_terms = len(terms)
    
    parent_child_pairs = []
    for i, term in enumerate(terms):
        ancestors = get_ancestors(term, go_graph)
        for anc in ancestors:
            if anc in term_to_idx:
                parent_child_pairs.append((term_to_idx[anc], i))
    print(f"  Terms: {n_terms}, Hierarchy pairs: {len(parent_child_pairs)}")
    
    y_train = np.zeros((len(train_accs), n_terms), dtype=np.float32)
    for acc, prot_terms in protein_to_terms.items():
        if acc in train_acc_to_idx:
            idx = train_acc_to_idx[acc]
            for j, term in enumerate(terms):
                if term in prot_terms:
                    y_train[idx, j] = 1
    
    print(f"  Positive labels: {int(y_train.sum()):,}")
    
    aspect_preds = np.zeros((len(test_accs), n_terms), dtype=np.float32)
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    strat = (y_train.sum(axis=1) > 0).astype(int)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, strat)):
        print(f"\n  Fold {fold+1}/{CONFIG['n_folds']}")
        
        model = WinnerStyleModel(
            X_train.shape[1], 
            CONFIG['hidden_dims'], 
            n_terms, 
            CONFIG['dropout'],
            CONFIG['residual']
        )
        
        # Use DataParallel for both GPUs during training
        if n_gpus >= 2:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
        criterion = HierarchicalLoss(parent_child_pairs, CONFIG['hierarchy_weight'])
        
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train[tr_idx], dtype=torch.float32),
                torch.tensor(y_train[tr_idx], dtype=torch.float32)
            ),
            batch_size=CONFIG['batch_size'] * n_gpus, shuffle=True
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        model.eval()
        test_loader = DataLoader(
            torch.tensor(X_test, dtype=torch.float32),
            batch_size=CONFIG['batch_size'] * n_gpus
        )
        
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
    
    for i, acc in enumerate(test_accs):
        for j, term in enumerate(terms):
            score = float(aspect_preds[i, j])
            if score > 0.01:
                all_predictions.append({'protein': acc, 'term': term, 'score': round(score, 6)})

# ============================================================================
# SUBMISSION
# ============================================================================

print("\n[5/6] Creating submission...")
df_preds = pd.DataFrame(all_predictions)
print(f"  Total: {len(df_preds):,}")

df_preds.to_csv('/kaggle/working/submission.tsv', sep='\t', index=False, header=False)

size_mb = os.path.getsize('/kaggle/working/submission.tsv') / 1024 / 1024
print(f"  Size: {size_mb:.1f} MB")

print("\n[6/6] Done!")
print("="*80)
print("🏆 DUAL-GPU SUBMISSION READY!")
print("="*80)
print(f"Expected F-max: 0.35-0.45")
