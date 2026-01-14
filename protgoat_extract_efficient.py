#!/usr/bin/env python3
"""
PROTGOAT-LITE: Memory-Efficient Embedding Extraction
=====================================================
Processes embeddings in smaller chunks to avoid crashes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧬 PROTGOAT-LITE: Memory-Efficient Embedding Extraction")
print("="*80)
print(f"Started: {datetime.now()}")

PROTGOAT_DIR = Path("/Users/hema/.cache/kagglehub/datasets/zmcxjt/cafa5-train-test-data/versions/2/")
OUTPUT_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Get CAFA-6 protein IDs
print("\n[1/4] Loading CAFA-6 protein IDs...")
train_ids = []
for record in SeqIO.parse("Train/train_sequences.fasta", "fasta"):
    protein_id = record.id.split('|')[0] if '|' in record.id else record.id
    train_ids.append(protein_id)
print(f"  Train: {len(train_ids):,} proteins")

test_ids = []
for record in SeqIO.parse("Test/testsuperset.fasta", "fasta"):
    protein_id = record.id.split('|')[0] if '|' in record.id else record.id
    test_ids.append(protein_id)
print(f"  Test: {len(test_ids):,} proteins")

train_set = set(train_ids)
test_set = set(test_ids)

# Step 2: Process ESM2-3B embeddings (memory-mapped)
print("\n[2/4] Loading ESM2-3B embeddings (memory-mapped)...")

# Use memory mapping to avoid loading all at once
esm2_test_emb = np.load(PROTGOAT_DIR / "ESM2_3B_test_embeddings_sorted.npy", mmap_mode='r')
esm2_test_labels = np.load(PROTGOAT_DIR / "ESM2_3B_test_labels_sorted.npy", allow_pickle=True)
print(f"  Test embeddings shape: {esm2_test_emb.shape}")

esm2_train_emb = np.load(PROTGOAT_DIR / "ESM2_3B_train_embeddings_sorted.npy", mmap_mode='r')
esm2_train_labels = np.load(PROTGOAT_DIR / "ESM2_3B_train_labels_sorted.npy", allow_pickle=True)
print(f"  Train embeddings shape: {esm2_train_emb.shape}")

# Step 3: Create lookup and extract
print("\n[3/4] Creating protein lookup...")

# Build lookup with array indices
protgoat_lookup = {}

# From PROTGOAT test set
for i, label in enumerate(esm2_test_labels):
    pid = label.split('|')[1] if '|' in str(label) else str(label)
    protgoat_lookup[pid] = ('test', i)

# From PROTGOAT train set (don't overwrite)
for i, label in enumerate(esm2_train_labels):
    pid = label.split('|')[1] if '|' in str(label) else str(label)
    if pid not in protgoat_lookup:
        protgoat_lookup[pid] = ('train', i)

print(f"  Total PROTGOAT proteins: {len(protgoat_lookup):,}")

# Step 4: Extract embeddings for CAFA-6
print("\n[4/4] Extracting embeddings for CAFA-6 proteins...")

EMB_DIM = 2560

# Extract CAFA-6 train embeddings
print("  Processing CAFA-6 train...")
cafa6_train_emb = np.zeros((len(train_ids), EMB_DIM), dtype=np.float32)
train_found = 0

for i, pid in enumerate(train_ids):
    if pid in protgoat_lookup:
        source, idx = protgoat_lookup[pid]
        if source == 'test':
            cafa6_train_emb[i] = esm2_test_emb[idx]
        else:
            cafa6_train_emb[i] = esm2_train_emb[idx]
        train_found += 1
    
    if (i + 1) % 20000 == 0:
        print(f"    {i+1:,}/{len(train_ids):,}")

print(f"  Train: {train_found}/{len(train_ids)} found ({100*train_found/len(train_ids):.1f}%)")

# Save train embeddings
np.save(OUTPUT_DIR / "cafa6_train_esm2.npy", cafa6_train_emb)
print(f"  Saved: cafa6_train_esm2.npy ({cafa6_train_emb.nbytes/1024/1024:.1f} MB)")
del cafa6_train_emb
gc.collect()

# Extract CAFA-6 test embeddings
print("\n  Processing CAFA-6 test...")
cafa6_test_emb = np.zeros((len(test_ids), EMB_DIM), dtype=np.float32)
test_found = 0

for i, pid in enumerate(test_ids):
    if pid in protgoat_lookup:
        source, idx = protgoat_lookup[pid]
        if source == 'test':
            cafa6_test_emb[i] = esm2_test_emb[idx]
        else:
            cafa6_test_emb[i] = esm2_train_emb[idx]
        test_found += 1
    
    if (i + 1) % 50000 == 0:
        print(f"    {i+1:,}/{len(test_ids):,}")

print(f"  Test: {test_found}/{len(test_ids)} found ({100*test_found/len(test_ids):.1f}%)")

# Save test embeddings
np.save(OUTPUT_DIR / "cafa6_test_esm2.npy", cafa6_test_emb)
print(f"  Saved: cafa6_test_esm2.npy ({cafa6_test_emb.nbytes/1024/1024:.1f} MB)")

# Save protein IDs
np.save(OUTPUT_DIR / "train_ids.npy", np.array(train_ids))
np.save(OUTPUT_DIR / "test_ids.npy", np.array(test_ids))

# Summary
print("\n" + "="*80)
print("✅ PROTGOAT-LITE EMBEDDINGS READY!")
print("="*80)
print(f"Train: {train_found}/{len(train_ids)} ({100*train_found/len(train_ids):.1f}%) with embeddings")
print(f"Test: {test_found}/{len(test_ids)} ({100*test_found/len(test_ids):.1f}%) with embeddings")
print(f"Missing: {len(test_ids) - test_found:,} proteins need new embeddings (~16 GPU hours)")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"Finished: {datetime.now()}")
