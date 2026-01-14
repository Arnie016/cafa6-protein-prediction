"""
Quick fix: Process DIAMOND results and create submission
"""
import pandas as pd
import numpy as np
import subprocess

print("="*80)
print("QUICK FIX: Processing DIAMOND results")
print("="*80)

# Load training annotations
print("\n[1/4] Loading training annotations...")
df_terms = pd.read_csv('Train/train_terms.tsv', sep='\t', 
                       names=['EntryID', 'term', 'aspect'])
protein_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
print(f"✅ Loaded annotations for {len(protein_terms):,} proteins")

# Load DIAMOND matches
print("\n[2/4] Loading DIAMOND matches...")
df_matches = pd.read_csv('/Volumes/TRANSCEND/cafa6_robust/predictions/diamond_matches.tsv', 
                         sep='\t', 
                         names=['query', 'subject', 'pident', 'evalue', 'bitscore'])

# Extract protein ID from subject (e.g. "sp|A0A0C5B5G6|MOTSC_HUMAN" -> "A0A0C5B5G6")
df_matches['subject_id'] = df_matches['subject'].str.split('|').str[1]
print(f"✅ Loaded {len(df_matches):,} matches for {df_matches['query'].nunique():,} proteins")

# Transfer annotations
print("\n[3/4] Transferring annotations...")
predictions = []

for query, group in df_matches.groupby('query'):
    term_scores = {}
    
    for _, hit in group.iterrows():
        subject = hit['subject_id']
        if subject in protein_terms:
            weight = hit['pident'] / 100.0
            for term in protein_terms[subject]:
                if term not in term_scores:
                    term_scores[term] = 0
                term_scores[term] = max(term_scores[term], weight)
    
    for term, score in term_scores.items():
        if score >= 0.01:
            predictions.append([query, term, round(score, 6)])

df_preds = pd.DataFrame(predictions, columns=['protein', 'term', 'score'])
print(f"✅ Generated {len(df_preds):,} predictions for {df_preds['protein'].nunique():,} proteins")

# Save
output_file = '/Volumes/TRANSCEND/cafa6_robust/predictions/diamond_predictions.tsv'
df_preds.to_csv(output_file, sep='\t', index=False, header=False)
print(f"✅ Saved: {output_file}")

# Apply GO propagation
print("\n[4/4] Applying GO hierarchy propagation...")
final_file = '/Volumes/TRANSCEND/cafa6_robust/predictions/final_submission.tsv'
subprocess.run([
    'python3', 'src/propagate_hierarchy.py',
    '--obo', 'Train/go-basic.obo',
    '--infile', output_file,
    '--outfile', final_file,
    '--min_score', '0.01'
], check=True)

# Stats
df_final = pd.read_csv(final_file, sep='\t', names=['protein', 'term', 'score'])
print(f"\n✅ COMPLETE!")
print(f"   Final predictions: {len(df_final):,}")
print(f"   Proteins covered: {df_final['protein'].nunique():,}")
print(f"   File: {final_file}")
