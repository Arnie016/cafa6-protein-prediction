#!/usr/bin/env python3
"""
PROTGOAT-LITE: Extract TEST embeddings only (train already done)
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧬 PROTGOAT-LITE: TEST EMBEDDINGS ONLY")
print("="*80)

PROTGOAT_DIR = Path("/Users/hema/.cache/kagglehub/datasets/zmcxjt/cafa5-train-test-data/versions/2/")
OUTPUT_DIR = Path("/Volumes/TRANSCEND/protgoat_lite")

def extract_accession(header):
    if '|' in header:
        return header.split('|')[1]
    return header

# Load test protein IDs
print("\n[1/4] Loading CAFA-6 test proteins...")
test_proteins = []
for record in SeqIO.parse("Test/testsuperset.fasta", "fasta"):
    test_proteins.append(extract_accession(record.id))
print(f"  Test proteins: {len(test_proteins):,}")

# Load PROTGOAT embeddings (memory-mapped)
print("\n[2/4] Loading PROTGOAT embeddings...")
esm2_test_emb = np.load(PROTGOAT_DIR / "ESM2_3B_test_embeddings_sorted.npy", mmap_mode='r')
esm2_test_labels = np.load(PROTGOAT_DIR / "ESM2_3B_test_labels_sorted.npy", allow_pickle=True)
esm2_train_emb = np.load(PROTGOAT_DIR / "ESM2_3B_train_embeddings_sorted.npy", mmap_mode='r')
esm2_train_labels = np.load(PROTGOAT_DIR / "ESM2_3B_train_labels_sorted.npy", allow_pickle=True)
print(f"  Loaded: test {esm2_test_emb.shape}, train {esm2_train_emb.shape}")

# Build lookup
print("\n[3/4] Building lookup...")
protgoat_lookup = {}
for i, label in enumerate(esm2_test_labels):
    protgoat_lookup[str(label)] = ('test', i)
for i, label in enumerate(esm2_train_labels):
    if str(label) not in protgoat_lookup:
        protgoat_lookup[str(label)] = ('train', i)
print(f"  PROTGOAT proteins: {len(protgoat_lookup):,}")

# Extract test embeddings
print("\n[4/4] Extracting test embeddings...")
EMB_DIM = 2560
cafa6_test_emb = np.zeros((len(test_proteins), EMB_DIM), dtype=np.float32)
test_found = 0

for i, acc in enumerate(test_proteins):
    if acc in protgoat_lookup:
        src, idx = protgoat_lookup[acc]
        cafa6_test_emb[i] = esm2_test_emb[idx] if src == 'test' else esm2_train_emb[idx]
        test_found += 1
    if (i + 1) % 50000 == 0:
        print(f"  {i+1:,}/{len(test_proteins):,} (found {test_found:,})")
        # Save checkpoint
        np.save(OUTPUT_DIR / f"test_checkpoint_{i+1}.npy", cafa6_test_emb[:i+1])

print(f"\n  Test: {test_found}/{len(test_proteins)} ({100*test_found/len(test_proteins):.1f}%)")

# Save final
np.save(OUTPUT_DIR / "cafa6_test_esm2.npy", cafa6_test_emb)
print(f"  Saved: cafa6_test_esm2.npy ({cafa6_test_emb.nbytes/1024/1024:.1f} MB)")

# Save accessions and missing list
test_accessions = np.array(test_proteins)
np.save(OUTPUT_DIR / "test_accessions.npy", test_accessions)

missing = [acc for acc in test_proteins if acc not in protgoat_lookup]
np.save(OUTPUT_DIR / "missing_accessions.npy", np.array(missing))

print(f"\n✅ COMPLETE!")
print(f"  Test coverage: {test_found}/{len(test_proteins)}")
print(f"  Missing: {len(missing):,}")
print(f"  Finished: {datetime.now()}")
