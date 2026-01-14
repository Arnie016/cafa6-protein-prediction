#!/usr/bin/env python3
"""
CAFA-6 FULL PIPELINE - GO ALL OUT!
===================================
Generates advanced features and prepares everything for Kaggle training.
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 CAFA-6 FULL ADVANCED FEATURE GENERATION")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
OUTPUT_DIR = Path("/Volumes/TRANSCEND/cafa6_robust/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Physicochemical properties
HYDROPHOBICITY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                  'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                  'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                  'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

MOLECULAR_WEIGHT = {'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
                    'Q': 146, 'E': 147, 'G': 75, 'H': 155, 'I': 131,
                    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
                    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117}

CHARGE = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
          'G': 0, 'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
          'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# Secondary structure propensity (Chou-Fasman)
HELIX = {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
         'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
         'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
         'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06}

SHEET = {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
         'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
         'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
         'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70}

def extract_features(seq):
    """Extract 440+ biologically meaningful features."""
    seq = ''.join(c for c in seq.upper() if c in AMINO_ACIDS)
    if len(seq) == 0:
        return None
    
    features = {}
    
    # 1. Amino acid composition (20 features)
    aa_counts = Counter(seq)
    for aa in AMINO_ACIDS:
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / len(seq)
    
    # 2. Dipeptide composition (400 features)
    dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]
    di_counts = Counter(dipeptides)
    total_di = len(dipeptides) if dipeptides else 1
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            features[f'di_{aa1}{aa2}'] = di_counts.get(aa1+aa2, 0) / total_di
    
    # 3. Physicochemical properties (10 features)
    hydro = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    features['hydro_mean'] = np.mean(hydro)
    features['hydro_std'] = np.std(hydro)
    features['hydro_max'] = max(hydro)
    features['hydro_min'] = min(hydro)
    
    mw = [MOLECULAR_WEIGHT.get(aa, 100) for aa in seq]
    features['mw_mean'] = np.mean(mw)
    features['mw_total'] = sum(mw)
    
    charges = [CHARGE.get(aa, 0) for aa in seq]
    features['charge_total'] = sum(charges)
    features['charge_mean'] = np.mean(charges)
    features['charge_positive'] = sum(1 for c in charges if c > 0) / len(seq)
    features['charge_negative'] = sum(1 for c in charges if c < 0) / len(seq)
    
    # 4. Secondary structure propensity (4 features)
    helix = [HELIX.get(aa, 1.0) for aa in seq]
    sheet = [SHEET.get(aa, 1.0) for aa in seq]
    features['helix_mean'] = np.mean(helix)
    features['sheet_mean'] = np.mean(sheet)
    features['helix_sheet_ratio'] = np.mean(helix) / (np.mean(sheet) + 0.001)
    features['struct_var'] = np.std(helix) + np.std(sheet)
    
    # 5. Length and complexity (10 features)
    features['length'] = len(seq)
    features['length_log'] = np.log1p(len(seq))
    
    # Sequence complexity (Shannon entropy)
    probs = [aa_counts.get(aa, 0)/len(seq) for aa in AMINO_ACIDS if aa_counts.get(aa, 0) > 0]
    features['entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Amino acid classes
    features['aromatic_frac'] = sum(1 for aa in seq if aa in 'FWY') / len(seq)
    features['aliphatic_frac'] = sum(1 for aa in seq if aa in 'AILV') / len(seq)
    features['polar_frac'] = sum(1 for aa in seq if aa in 'STNQ') / len(seq)
    features['tiny_frac'] = sum(1 for aa in seq if aa in 'AGS') / len(seq)
    features['cysteine_frac'] = seq.count('C') / len(seq)
    features['proline_frac'] = seq.count('P') / len(seq)
    features['glycine_frac'] = seq.count('G') / len(seq)
    
    return features

def process_fasta(fasta_path, output_path, dataset_name):
    """Process a FASTA file and save features."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    print(f"Found {len(records):,} sequences")
    
    all_features = []
    for i, record in enumerate(records):
        protein_id = record.id.split('|')[0] if '|' in record.id else record.id
        
        feats = extract_features(str(record.seq))
        if feats:
            feats['ProteinID'] = protein_id
            all_features.append(feats)
        
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1:,}/{len(records):,}")
    
    df = pd.DataFrame(all_features)
    
    # Reorder columns
    cols = ['ProteinID'] + [c for c in df.columns if c != 'ProteinID']
    df = df[cols]
    
    # Save
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Saved {len(df):,} proteins with {len(df.columns)-1} features")
    print(f"   File: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return df

# Main execution
if __name__ == "__main__":
    # Process training data
    df_train = process_fasta(
        'Train/train_sequences.fasta',
        OUTPUT_DIR / 'train_advanced.parquet',
        'Training Set'
    )
    
    # Process test data
    df_test = process_fasta(
        'Test/testsuperset.fasta',
        OUTPUT_DIR / 'test_advanced.parquet',
        'Test Set'
    )
    
    print("\n" + "="*80)
    print("🎉 FEATURE GENERATION COMPLETE!")
    print("="*80)
    print(f"\nFeature dimensions: {len(df_train.columns)-1}")
    print(f"Training proteins: {len(df_train):,}")
    print(f"Test proteins: {len(df_test):,}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
