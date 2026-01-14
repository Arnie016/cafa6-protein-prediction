"""
Transfer GO terms from 3Di structural homology matches.

Usage:
    python transfer_3di_go.py

Input:
    - 3di_matches.tsv (DIAMOND output from 3Di alignment)
    - Train/train_terms.tsv (GO annotations for training proteins)

Output:
    - submission_3di.tsv (GO predictions based on structural homology)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def main():
    print("=" * 80)
    print("🧬 3Di Structural Homology GO Transfer")
    print("=" * 80)
    
    # Load 3Di matches (DIAMOND output format 6)
    print("\n[1/4] Loading 3Di matches...")
    cols = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    
    try:
        matches = pd.read_csv("3di_matches.tsv", sep="\t", names=cols)
        print(f"  Loaded {len(matches)} matches")
    except FileNotFoundError:
        print("ERROR: 3di_matches.tsv not found!")
        print("Run DIAMOND first:")
        print("  diamond makedb --in train_3di.fasta -d train_3di_db")
        print("  diamond blastp -q test_3di.fasta -d train_3di_db --custom-matrix 3di_matrix.mat -o 3di_matches.tsv")
        return
    
    # Load training GO terms
    print("\n[2/4] Loading training GO annotations...")
    train_terms = pd.read_csv("Train/train_terms.tsv", sep="\t")
    
    # Build protein -> GO terms mapping
    train_go = defaultdict(set)
    for _, row in train_terms.iterrows():
        train_go[row['EntryID']].add(row['term'])
    print(f"  Loaded GO terms for {len(train_go)} proteins")
    
    # Transfer GO terms
    print("\n[3/4] Transferring GO terms from structural homologs...")
    
    predictions = defaultdict(lambda: defaultdict(float))  # protein -> term -> score
    
    for _, hit in tqdm(matches.iterrows(), total=len(matches)):
        query = hit['qseqid']
        target = hit['sseqid']
        bitscore = hit['bitscore']
        
        # Normalize bitscore to 0-1 range (roughly)
        # Typical max bitscore ~1000, min ~50
        score = min(1.0, max(0.0, bitscore / 500.0))
        
        # Transfer all GO terms from the hit
        if target in train_go:
            for term in train_go[target]:
                # Take max score across all hits
                predictions[query][term] = max(predictions[query][term], score)
    
    print(f"  Generated predictions for {len(predictions)} proteins")
    
    # Write output
    print("\n[4/4] Writing submission file...")
    
    rows = []
    for protein, terms in predictions.items():
        for term, score in terms.items():
            if score >= 0.01:  # Filter very low scores
                rows.append((protein, term, f"{score:.6f}"))
    
    df = pd.DataFrame(rows, columns=['protein', 'term', 'score'])
    df.to_csv("submission_3di.tsv", sep="\t", index=False, header=False)
    
    print(f"\n✅ Done! Wrote {len(rows)} predictions to submission_3di.tsv")
    print("\nThis file can be added to your ensemble!")

if __name__ == "__main__":
    main()
