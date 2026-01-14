#!/usr/bin/env python3
"""
Fetch UniProt descriptions for CAFA-6 proteins
Saves to a TSV file that can be uploaded to Kaggle
"""

import requests
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import time
import sys

print("="*60)
print("📥 Fetching UniProt Descriptions")
print("="*60)

# Load protein accessions
train_accs = set()
for record in SeqIO.parse("Train/train_sequences.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    train_accs.add(acc)

test_accs = set()
for record in SeqIO.parse("Test/testsuperset.fasta", "fasta"):
    acc = record.id.split('|')[1] if '|' in record.id else record.id
    test_accs.add(acc)

all_accs = list(train_accs | test_accs)
print(f"Total proteins: {len(all_accs):,}")

# Fetch from UniProt API
descriptions = {}

def fetch_batch(accs, batch_num, total_batches):
    """Fetch descriptions for a batch of accessions."""
    query = " OR ".join([f"accession:{acc}" for acc in accs])
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,protein_name,cc_function",
        "format": "tsv",
        "size": 500
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 2:
                    acc = parts[0]
                    name = parts[1] if len(parts) > 1 else ""
                    func = parts[2] if len(parts) > 2 else ""
                    descriptions[acc] = f"{name}. {func}"[:500]
        print(f"  Batch {batch_num}/{total_batches}: {len(descriptions)} total")
    except Exception as e:
        print(f"  Batch {batch_num} error: {e}")
    
    time.sleep(0.5)  # Rate limit

# Process in batches
batch_size = 50
batches = [all_accs[i:i+batch_size] for i in range(0, len(all_accs), batch_size)]
total_batches = len(batches)

print(f"\nFetching in {total_batches} batches...")
for i, batch in enumerate(batches):
    fetch_batch(batch, i+1, total_batches)
    
    # Save checkpoint every 100 batches
    if (i+1) % 100 == 0:
        df = pd.DataFrame([
            {"accession": acc, "description": desc} 
            for acc, desc in descriptions.items()
        ])
        df.to_csv("uniprot_descriptions_checkpoint.tsv", sep='\t', index=False)
        print(f"  Checkpoint saved: {len(descriptions)} descriptions")

# Save final
df = pd.DataFrame([
    {"accession": acc, "description": desc} 
    for acc, desc in descriptions.items()
])
df.to_csv("uniprot_descriptions.tsv", sep='\t', index=False)

print(f"\n✅ Done! Saved {len(descriptions)} descriptions to uniprot_descriptions.tsv")
print(f"   Coverage: {len(descriptions)}/{len(all_accs)} ({100*len(descriptions)/len(all_accs):.1f}%)")
