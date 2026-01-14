#!/usr/bin/env python3
"""
PROTGOAT-LITE: Fixed ID Matching
=================================
Correctly extracts accession IDs from UniProt format.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧬 PROTGOAT-LITE: Embedding Extraction (Fixed IDs)")
print("="*80)
print(f"Started: {datetime.now()}")

PROTGOAT_DIR = Path("/Users/hema/.cache/kagglehub/datasets/zmcxjt/cafa5-train-test-data/versions/2/")
OUTPUT_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_accession(header):
    """Extract UniProt accession from various formats."""
    # Format: sp|A0A0C5B5G6|MOTSC_HUMAN -> A0A0C5B5G6
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    return header

# Step 1: Get CAFA-6 protein IDs
print("\n[1/4] Loading CAFA-6 protein IDs...")
train_ids = []
train_accessions = []
for record in SeqIO.parse("Train/train_sequences.fasta", "fasta"):
    train_ids.append(record.id)
    train_accessions.append(extract_accession(record.id))
print(f"  Train: {len(train_ids):,} proteins")
print(f"  Sample: {train_ids[0]} -> {train_accessions[0]}")

test_ids = []
test_accessions = []
for record in SeqIO.parse("Test/testsuperset.fasta", "fasta"):
    test_ids.append(record.id)
    test_accessions.append(extract_accession(record.id))
print(f"  Test: {len(test_ids):,} proteins")
print(f"  Sample: {test_ids[0]} -> {test_accessions[0]}")

# Step 2: Load PROTGOAT embeddings (memory-mapped)
print("\n[2/4] Loading ESM2-3B embeddings (memory-mapped)...")
esm2_test_emb = np.load(PROTGOAT_DIR / "ESM2_3B_test_embeddings_sorted.npy", mmap_mode='r')
esm2_test_labels = np.load(PROTGOAT_DIR / "ESM2_3B_test_labels_sorted.npy", allow_pickle=True)
print(f"  PROTGOAT test: {esm2_test_emb.shape}")

esm2_train_emb = np.load(PROTGOAT_DIR / "ESM2_3B_train_embeddings_sorted.npy", mmap_mode='r')
esm2_train_labels = np.load(PROTGOAT_DIR / "ESM2_3B_train_labels_sorted.npy", allow_pickle=True)
print(f"  PROTGOAT train: {esm2_train_emb.shape}")

# Step 3: Build lookup by accession
print("\n[3/4] Building accession-to-embedding lookup...")
protgoat_lookup = {}

for i, label in enumerate(esm2_test_labels):
    accession = str(label).strip()
    protgoat_lookup[accession] = ('test', i)

for i, label in enumerate(esm2_train_labels):
    accession = str(label).strip()
    if accession not in protgoat_lookup:
        protgoat_lookup[accession] = ('train', i)

print(f"  PROTGOAT accessions: {len(protgoat_lookup):,}")
print(f"  Sample: {list(protgoat_lookup.keys())[:3]}")

# Check overlap
train_overlap = sum(1 for acc in train_accessions if acc in protgoat_lookup)
test_overlap = sum(1 for acc in test_accessions if acc in protgoat_lookup)
print(f"  Train overlap: {train_overlap}/{len(train_accessions)} ({100*train_overlap/len(train_accessions):.1f}%)")
print(f"  Test overlap: {test_overlap}/{len(test_accessions)} ({100*test_overlap/len(test_accessions):.1f}%)")

# Step 4: Extract embeddings
print("\n[4/4] Extracting embeddings...")

EMB_DIM = 2560

# Train embeddings
print("  Processing train...")
cafa6_train_emb = np.zeros((len(train_accessions), EMB_DIM), dtype=np.float32)
train_found = 0

for i, acc in enumerate(train_accessions):
    if acc in protgoat_lookup:
        source, idx = protgoat_lookup[acc]
        if source == 'test':
            cafa6_train_emb[i] = esm2_test_emb[idx]
        else:
            cafa6_train_emb[i] = esm2_train_emb[idx]
        train_found += 1
    if (i + 1) % 20000 == 0:
        print(f"    {i+1:,}/{len(train_accessions):,} (found {train_found:,})")

print(f"  Train: {train_found}/{len(train_accessions)} ({100*train_found/len(train_accessions):.1f}%)")
np.save(OUTPUT_DIR / "cafa6_train_esm2.npy", cafa6_train_emb)
print(f"  Saved: cafa6_train_esm2.npy ({cafa6_train_emb.nbytes/1024/1024:.1f} MB)")
del cafa6_train_emb
gc.collect()

# Test embeddings  
print("\n  Processing test...")
cafa6_test_emb = np.zeros((len(test_accessions), EMB_DIM), dtype=np.float32)
test_found = 0

for i, acc in enumerate(test_accessions):
    if acc in protgoat_lookup:
        source, idx = protgoat_lookup[acc]
        if source == 'test':
            cafa6_test_emb[i] = esm2_test_emb[idx]
        else:
            cafa6_test_emb[i] = esm2_train_emb[idx]
        test_found += 1
    if (i + 1) % 50000 == 0:
        print(f"    {i+1:,}/{len(test_accessions):,} (found {test_found:,})")

print(f"  Test: {test_found}/{len(test_accessions)} ({100*test_found/len(test_accessions):.1f}%)")
np.save(OUTPUT_DIR / "cafa6_test_esm2.npy", cafa6_test_emb)
print(f"  Saved: cafa6_test_esm2.npy ({cafa6_test_emb.nbytes/1024/1024:.1f} MB)")

# Save IDs
np.save(OUTPUT_DIR / "train_ids.npy", np.array(train_ids))
np.save(OUTPUT_DIR / "test_ids.npy", np.array(test_ids))
np.save(OUTPUT_DIR / "train_accessions.npy", np.array(train_accessions))
np.save(OUTPUT_DIR / "test_accessions.npy", np.array(test_accessions))

# Track which proteins need new embeddings
missing_test = [acc for acc in test_accessions if acc not in protgoat_lookup]
np.save(OUTPUT_DIR / "missing_test_accessions.npy", np.array(missing_test))
print(f"\n  Missing (need generation): {len(missing_test):,} proteins")

print("\n" + "="*80)
print("✅ PROTGOAT EMBEDDINGS EXTRACTED!")
print("="*80)
print(f"Train coverage: {train_found}/{len(train_accessions)} ({100*train_found/len(train_accessions):.1f}%)")
print(f"Test coverage: {test_found}/{len(test_accessions)} ({100*test_found/len(test_accessions):.1f}%)")
print(f"Missing (need Kaggle): {len(missing_test):,} proteins")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"Finished: {datetime.now()}")
