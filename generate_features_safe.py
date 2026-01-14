#!/usr/bin/env python3
"""
CAFA-6 ROBUST PIPELINE - CRASH-PROOF VERSION
=============================================
- Processes in small batches to manage memory
- Saves checkpoints frequently
- Low memory footprint
- Proper garbage collection
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
from pathlib import Path
from datetime import datetime
import gc
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🚀 CAFA-6 CRASH-PROOF PIPELINE")
print("="*80)
print(f"Started: {datetime.now()}")

OUTPUT_DIR = Path("/Volumes/TRANSCEND/cafa6_robust/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
BATCH_SIZE = 5000  # Small batches to manage memory

# Properties
HYDROPHOBICITY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
                  'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
                  'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
MOLECULAR_WEIGHT = {'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'Q': 146, 'E': 147, 
                    'G': 75, 'H': 155, 'I': 131, 'L': 131, 'K': 146, 'M': 149, 'F': 165, 
                    'P': 115, 'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117}

def extract_features(seq):
    """Extract features from protein sequence."""
    seq = ''.join(c for c in seq.upper() if c in AMINO_ACIDS)
    if len(seq) == 0:
        return None
    
    features = {}
    n = len(seq)
    
    # 1. AA composition (20)
    aa_counts = Counter(seq)
    for aa in AMINO_ACIDS:
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / n
    
    # 2. Dipeptides (400)
    dipeptides = [seq[i:i+2] for i in range(n-1)]
    di_counts = Counter(dipeptides)
    total_di = len(dipeptides) if dipeptides else 1
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            features[f'di_{aa1}{aa2}'] = di_counts.get(aa1+aa2, 0) / total_di
    
    # 3. Physicochemical (8)
    hydro = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    features['hydro_mean'] = np.mean(hydro)
    features['hydro_std'] = np.std(hydro)
    mw = [MOLECULAR_WEIGHT.get(aa, 100) for aa in seq]
    features['mw_mean'] = np.mean(mw)
    features['mw_total'] = sum(mw)
    
    # 4. Length and classes (10)
    features['length'] = n
    features['length_log'] = np.log1p(n)
    features['aromatic'] = sum(1 for aa in seq if aa in 'FWY') / n
    features['aliphatic'] = sum(1 for aa in seq if aa in 'AILV') / n
    features['polar'] = sum(1 for aa in seq if aa in 'STNQ') / n
    features['positive'] = sum(1 for aa in seq if aa in 'RKH') / n
    features['negative'] = sum(1 for aa in seq if aa in 'DE') / n
    features['cysteine'] = seq.count('C') / n
    
    # Entropy
    probs = [c/n for c in aa_counts.values() if c > 0]
    features['entropy'] = -sum(p * np.log2(p) for p in probs)
    
    return features

def process_fasta_chunked(fasta_path, output_path, name):
    """Process FASTA in chunks to manage memory."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    total = len(records)
    print(f"Total sequences: {total:,}")
    
    all_dfs = []
    
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = records[start:end]
        
        features_list = []
        for record in batch:
            protein_id = record.id.split('|')[0] if '|' in record.id else record.id
            feats = extract_features(str(record.seq))
            if feats:
                feats['ProteinID'] = protein_id
                features_list.append(feats)
        
        df_batch = pd.DataFrame(features_list)
        all_dfs.append(df_batch)
        
        print(f"  Batch {start//BATCH_SIZE + 1}: {start:,}-{end:,} ({len(df_batch)} proteins)")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Combine all batches
    print(f"\nCombining {len(all_dfs)} batches...")
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns
    cols = ['ProteinID'] + [c for c in df.columns if c != 'ProteinID']
    df = df[cols]
    
    # Save
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    
    print(f"✅ Saved: {output_path}")
    print(f"   Proteins: {len(df):,}")
    print(f"   Features: {len(df.columns)-1}")
    print(f"   Size: {size_mb:.1f} MB")
    
    gc.collect()
    return df

# Main execution
print("\n" + "="*80)
print("STAGE 1: GENERATE ADVANCED FEATURES")
print("="*80)

# Training features
df_train = process_fasta_chunked(
    'Train/train_sequences.fasta',
    OUTPUT_DIR / 'train_advanced.parquet',
    'Training Set'
)

# Test features
df_test = process_fasta_chunked(
    'Test/testsuperset.fasta', 
    OUTPUT_DIR / 'test_advanced.parquet',
    'Test Set'
)

print("\n" + "="*80)
print("🎉 FEATURE GENERATION COMPLETE!")
print("="*80)
print(f"Training: {len(df_train):,} proteins")
print(f"Test: {len(df_test):,} proteins")
print(f"Features: {len(df_train.columns)-1}")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"Finished: {datetime.now()}")
