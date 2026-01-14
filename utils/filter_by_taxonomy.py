#!/usr/bin/env python3
"""
Taxonomy Filtering for CAFA-6
=============================
Filters predictions based on species-specific GO terms observed in training data.

Logic:
1. Identify the species (Taxon ID) of each test protein.
2. Check if that species exists in Training Data.
3. If species has sufficient training data (>100 proteins), RESTRICT predictions 
   to only GO terms (and their ancestors) observed in that species.

This removes biologically impossible terms (e.g., "photosynthesis" in humans).
"""

import pandas as pd
import obonet
import networkx as nx
import argparse
from collections import defaultdict
import sys
import os

def load_go_graph(obo_path):
    print(f"Loading GO hierarchy from {obo_path}...")
    return obonet.read_obo(obo_path)

def get_ancestors(term, graph):
    ancestors = set()
    if term in graph:
        try:
            for parent in graph.predecessors(term):
                ancestors.add(parent)
                ancestors.update(get_ancestors(parent, graph))
        except:
            pass
    return ancestors

def main():
    parser = argparse.ArgumentParser(description="Filter CAFA-6 predictions by taxonomy")
    parser.add_argument("--preds", required=True, help="Input prediction TSV file")
    parser.add_argument("--out", required=True, help="Output filtered TSV file")
    parser.add_argument("--train-taxonomy", default="Train/train_taxonomy.tsv", help="Train taxonomy file")
    parser.add_argument("--train-terms", default="Train/train_terms.tsv", help="Train terms file")
    parser.add_argument("--test-taxonomy", default="Test/testsuperset-taxon-list.tsv", help="Test taxonomy file")
    parser.add_argument("--obo", default="Train/go-basic.obo", help="GO OBO file")
    parser.add_argument("--min-proteins", type=int, default=50, help="Min training proteins to apply filter")
    args = parser.parse_args()

    # 1. Load Taxonomy Mappings
    print("Loading taxonomy data...")
    # Train: protein -> taxa
    df_train_tax = pd.read_csv(args.train_taxonomy, sep='\t')
    train_prot_to_tax = dict(zip(df_train_tax.iloc[:,0], df_train_tax.iloc[:,1]))
    
    # Test: protein -> taxa (from FASTA headers)
    print(f"Parsing test taxonomy from {args.test_taxonomy}...")
    test_prot_to_tax = {}
    with open(args.test_taxonomy, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    try:
                        test_prot_to_tax[parts[0]] = int(parts[1])
                    except ValueError:
                        pass

    
    print(f"  Train taxonomy: {len(train_prot_to_tax):,} proteins")
    print(f"  Test taxonomy:  {len(test_prot_to_tax):,} proteins")

    # 2. Build Allowed Terms per Species from Training
    print("Building allowed terms per species...")
    df_terms = pd.read_csv(args.train_terms, sep='\t', names=['protein', 'term', 'aspect'])
    if df_terms.iloc[0,0] == 'EntryID': df_terms = df_terms.iloc[1:] # Skip header
    
    # Group terms by protein
    prot_to_terms = df_terms.groupby('protein')['term'].apply(set).to_dict()
    
    # Map Taxon -> Set(Terms)
    taxon_terms = defaultdict(set)
    taxon_protein_counts = defaultdict(set)
    
    for prot, terms in prot_to_terms.items():
        if prot in train_prot_to_tax:
            tax = train_prot_to_tax[prot]
            taxon_terms[tax].update(terms)
            taxon_protein_counts[tax].add(prot)
            
    taxon_counts = {k: len(v) for k, v in taxon_protein_counts.items()}
    print(f"  Found {len(taxon_terms)} species in training.")
    
    # 3. Propagate Allowed Terms (Add Ancestors)
    go_graph = load_go_graph(args.obo)
    print("Propagating allowed terms (adding ancestors)...")
    
    final_allowed_terms = {}
    
    for tax, terms in taxon_terms.items():
        count = taxon_counts[tax]
        if count < args.min_proteins:
            continue
            
        # Add ancestors for all observed terms
        expanded_terms = set(terms)
        for term in terms:
            expanded_terms.update(get_ancestors(term, go_graph))
            
        final_allowed_terms[tax] = expanded_terms
        
    print(f"  Filtering active for {len(final_allowed_terms)} species (with >{args.min_proteins} proteins).")

    # 4. Filter Predictions
    print(f"Filtering predictions from {args.preds}...")
    
    filtered_count = 0
    total_count = 0
    
    with open(args.preds, 'r') as bin, open(args.out, 'w') as bout:
        for line in bin:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            prot, term, score = parts[0], parts[1], float(parts[2])
            total_count += 1
            
            # Check if we should filter
            if prot in test_prot_to_tax:
                tax = test_prot_to_tax[prot]
                
                # If we have a filter for this species
                if tax in final_allowed_terms:
                    # If term is NOT allowed (not seen in training), DROP IT
                    if term not in final_allowed_terms[tax]:
                        filtered_count += 1
                        continue 
            
            bout.write(line)
            
    print("="*60)
    print("✅ FILTERING COMPLETE")
    print(f"  Input predictions: {total_count:,}")
    print(f"  Filtered out:      {filtered_count:,} ({100*filtered_count/total_count:.1f}%)")
    print(f"  Output file:       {args.out}")
    print("="*60)

if __name__ == "__main__":
    main()
