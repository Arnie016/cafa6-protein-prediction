#!/usr/bin/env python3
"""
PROTGOAT-LITE IMPLEMENTATION
=============================
Based on CAFA-5 4th place solution (0.56 F-max)

Key innovations implemented:
1. Multi-PLM embeddings (ESM2-3B + Ankh)
2. Text embeddings from abstracts
3. Neural network with GO-DAG awareness
4. Simplified training (6 models vs 25)

This script:
1. Loads PROTGOAT embeddings for overlapping proteins
2. Creates embedding lookup for CAFA-6 proteins
3. Prepares training data for neural network
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import gc

print("="*80)
print("🧬 PROTGOAT-LITE: Stage 1 - Embedding Preparation")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
PROTGOAT_DIR = Path("/Users/hema/.cache/kagglehub/datasets/zmcxjt/cafa5-train-test-data/versions/2/")
CAFA6_DATA = Path("/Volumes/TRANSCEND/cafa6_data")
OUTPUT_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load protein overlap analysis
print("\n[1/5] Loading protein overlap analysis...")
proteins_with_protgoat = np.load(CAFA6_DATA / "cafa6_proteins_with_protgoat.npy", allow_pickle=True)
proteins_need_embeddings = np.load(CAFA6_DATA / "cafa6_proteins_need_embeddings.npy", allow_pickle=True)

print(f"  Proteins with PROTGOAT embeddings: {len(proteins_with_protgoat):,}")
print(f"  Proteins needing new embeddings: {len(proteins_need_embeddings):,}")

# Load PROTGOAT ESM2-3B embeddings
print("\n[2/5] Loading ESM2-3B embeddings from PROTGOAT...")
esm2_test_embeddings = np.load(PROTGOAT_DIR / "ESM2_3B_test_embeddings_sorted.npy")
esm2_test_labels = np.load(PROTGOAT_DIR / "ESM2_3B_test_labels_sorted.npy", allow_pickle=True)
print(f"  ESM2-3B test: {esm2_test_embeddings.shape}")

esm2_train_embeddings = np.load(PROTGOAT_DIR / "ESM2_3B_train_embeddings_sorted.npy")
esm2_train_labels = np.load(PROTGOAT_DIR / "ESM2_3B_train_labels_sorted.npy", allow_pickle=True)
print(f"  ESM2-3B train: {esm2_train_embeddings.shape}")

# Create protein -> embedding lookup
print("\n[3/5] Creating protein-to-embedding lookup...")
esm2_lookup = {}

# From test set
for i, label in enumerate(esm2_test_labels):
    protein_id = label.split('|')[1] if '|' in str(label) else str(label)
    esm2_lookup[protein_id] = esm2_test_embeddings[i]

# From train set
for i, label in enumerate(esm2_train_labels):
    protein_id = label.split('|')[1] if '|' in str(label) else str(label)
    if protein_id not in esm2_lookup:  # Don't overwrite
        esm2_lookup[protein_id] = esm2_train_embeddings[i]

print(f"  Total proteins in lookup: {len(esm2_lookup):,}")

# Free memory
del esm2_test_embeddings, esm2_train_embeddings
gc.collect()

# Load Ankh embeddings 
print("\n[4/5] Loading Ankh embeddings...")
try:
    ankh_test_embeddings = np.load(PROTGOAT_DIR / "Ankh_test_embeddings_sorted.npy")
    ankh_test_labels = np.load(PROTGOAT_DIR / "Ankh_test_labels_sorted.npy", allow_pickle=True)
    print(f"  Ankh test: {ankh_test_embeddings.shape}")
    
    ankh_train_embeddings = np.load(PROTGOAT_DIR / "Ankh_train_embeddings_sorted.npy")
    ankh_train_labels = np.load(PROTGOAT_DIR / "Ankh_train_labels_sorted.npy", allow_pickle=True)
    print(f"  Ankh train: {ankh_train_embeddings.shape}")
    
    ankh_lookup = {}
    for i, label in enumerate(ankh_test_labels):
        protein_id = label.split('|')[1] if '|' in str(label) else str(label)
        ankh_lookup[protein_id] = ankh_test_embeddings[i]
    for i, label in enumerate(ankh_train_labels):
        protein_id = label.split('|')[1] if '|' in str(label) else str(label)
        if protein_id not in ankh_lookup:
            ankh_lookup[protein_id] = ankh_train_embeddings[i]
    
    print(f"  Ankh proteins in lookup: {len(ankh_lookup):,}")
    del ankh_test_embeddings, ankh_train_embeddings
    gc.collect()
    
except Exception as e:
    print(f"  Ankh loading failed: {e}")
    ankh_lookup = {}

# Extract embeddings for CAFA-6 proteins
print("\n[5/5] Extracting embeddings for CAFA-6 proteins...")

# Load CAFA-6 train protein IDs
train_ids = []
from Bio import SeqIO
for record in SeqIO.parse("Train/train_sequences.fasta", "fasta"):
    protein_id = record.id.split('|')[0] if '|' in record.id else record.id
    train_ids.append(protein_id)
print(f"  CAFA-6 train proteins: {len(train_ids):,}")

# Load CAFA-6 test protein IDs
test_ids = []
for record in SeqIO.parse("Test/testsuperset.fasta", "fasta"):
    protein_id = record.id.split('|')[0] if '|' in record.id else record.id
    test_ids.append(protein_id)
print(f"  CAFA-6 test proteins: {len(test_ids):,}")

# Extract train embeddings
print("\n  Extracting train embeddings...")
train_esm2 = []
train_found = 0
for pid in train_ids:
    if pid in esm2_lookup:
        train_esm2.append(esm2_lookup[pid])
        train_found += 1
    else:
        # Zero vector for missing (will be replaced later)
        train_esm2.append(np.zeros(2560, dtype=np.float32))

train_esm2 = np.array(train_esm2, dtype=np.float32)
print(f"    Train: {train_found}/{len(train_ids)} found ({100*train_found/len(train_ids):.1f}%)")
print(f"    Shape: {train_esm2.shape}")

# Extract test embeddings
print("\n  Extracting test embeddings...")
test_esm2 = []
test_found = 0
for pid in test_ids:
    if pid in esm2_lookup:
        test_esm2.append(esm2_lookup[pid])
        test_found += 1
    else:
        test_esm2.append(np.zeros(2560, dtype=np.float32))

test_esm2 = np.array(test_esm2, dtype=np.float32)
print(f"    Test: {test_found}/{len(test_ids)} found ({100*test_found/len(test_ids):.1f}%)")
print(f"    Shape: {test_esm2.shape}")

# Save embeddings
print("\n[SAVING] Saving extracted embeddings...")
np.save(OUTPUT_DIR / "cafa6_train_esm2_embeddings.npy", train_esm2)
np.save(OUTPUT_DIR / "cafa6_test_esm2_embeddings.npy", test_esm2)
np.save(OUTPUT_DIR / "cafa6_train_ids.npy", np.array(train_ids))
np.save(OUTPUT_DIR / "cafa6_test_ids.npy", np.array(test_ids))

print(f"\n✅ Saved to: {OUTPUT_DIR}")
print(f"   cafa6_train_esm2_embeddings.npy: {train_esm2.nbytes/1024/1024:.1f} MB")
print(f"   cafa6_test_esm2_embeddings.npy: {test_esm2.nbytes/1024/1024:.1f} MB")

# Summary
print("\n" + "="*80)
print("📊 EMBEDDING EXTRACTION SUMMARY")
print("="*80)
print(f"Train coverage: {train_found}/{len(train_ids)} ({100*train_found/len(train_ids):.1f}%)")
print(f"Test coverage: {test_found}/{len(test_ids)} ({100*test_found/len(test_ids):.1f}%)")
print(f"Embedding dimension: 2560 (ESM2-3B)")
print(f"\nNext step: Train neural network on these embeddings")
print(f"Expected F-max: 0.35-0.40")
print(f"Finished: {datetime.now()}")
