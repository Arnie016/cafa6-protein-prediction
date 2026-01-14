#!/usr/bin/env python3
"""
Ensemble Stacking for CAFA-6
============================
Combines predictions from multiple models using Weighted Averaging.

Usage:
    python3 ensemble_predictions.py --inputs file1.tsv file2.tsv --weights 0.5 0.5 --output ensemble.tsv

Logic:
1. Reads protein-term-score triplets from all files.
2. Aligns predictions by (Protein, Term).
3. Computes final score = Sum(Weight_i * Score_i) / Sum(Weights).
4. Saves to output file.
"""

import pandas as pd
import argparse
import sys
from collections import defaultdict
import gc

def main():
    parser = argparse.ArgumentParser(description="Ensemble CAFA-6 predictions")
    parser.add_argument("--inputs", required=True, nargs='+', help="Input prediction TSV files")
    parser.add_argument("--weights", required=True, nargs='+', type=float, help="Weights for each file")
    parser.add_argument("--output", required=True, help="Output ensemble TSV file")
    args = parser.parse_args()

    if len(args.inputs) != len(args.weights):
        print("Error: Number of inputs must match number of weights!")
        sys.exit(1)

    print("="*60)
    print("🧬 CAFA-6 Ensemble Stacking")
    print("="*60)
    
    # Store aggregated scores: {(protein, term): sum_weighted_score}
    combined_scores = defaultdict(float)
    
    # Validation check: Normalize weights
    total_weight = sum(args.weights)
    norm_weights = [w / total_weight for w in args.weights]
    
    print(f"Models: {len(args.inputs)}")
    print(f"Weights: {norm_weights}")
    
    # Process each file
    for i, (fname, weight) in enumerate(zip(args.inputs, norm_weights)):
        print(f"\n[{i+1}/{len(args.inputs)}] Reading {fname} (Weight: {weight:.2f})...")
        
        count = 0
        with open(fname, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        prot, term, score = parts[0], parts[1], float(parts[2])
                        # Accumulate weighted score
                        combined_scores[(prot, term)] += score * weight
                        count += 1
                    except ValueError:
                        pass
                        
        print(f"  Loaded {count:,} predictions.")
        gc.collect()

    print(f"\nSaving {len(combined_scores):,} combined predictions to {args.output}...")
    
    with open(args.output, 'w') as out:
        for (prot, term), score in combined_scores.items():
            # Round to 6 decimals to save space
            out.write(f"{prot}\t{term}\t{round(score, 6)}\n")
            
    print("\n✅ Ensemble Complete!")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
