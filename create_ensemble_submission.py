"""
CAFA-6 ENSEMBLE SUBMISSION CREATOR
===================================
Combines DIAMOND homology + LightGBM advanced features predictions
and applies GO hierarchy propagation.
"""

import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

print("="*80)
print("🚀 CAFA-6 ENSEMBLE SUBMISSION CREATOR")
print("="*80)

OUTPUT_DIR = Path("/Volumes/TRANSCEND/cafa6_robust")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

def create_ensemble():
    """Create ensemble from multiple prediction sources."""
    
    # Load DIAMOND predictions
    print("\n[1/4] Loading DIAMOND predictions...")
    diamond_file = PREDICTIONS_DIR / "diamond_predictions.tsv"
    df_diamond = pd.read_csv(diamond_file, sep='\t', names=['protein', 'term', 'score'])
    print(f"  DIAMOND: {len(df_diamond):,} predictions")
    
    # Check if ML predictions exist
    ml_file = PREDICTIONS_DIR / "ml_advanced_predictions.tsv"
    if ml_file.exists():
        print("\n[2/4] Loading ML predictions...")
        df_ml = pd.read_csv(ml_file, sep='\t', names=['protein', 'term', 'score'])
        print(f"  ML: {len(df_ml):,} predictions")
        
        # Weighted ensemble
        print("\n[3/4] Creating weighted ensemble...")
        # DIAMOND weight: 0.6, ML weight: 0.4
        df_diamond['weight'] = 0.6
        df_ml['weight'] = 0.4
        
        df_combined = pd.concat([df_diamond, df_ml])
        df_ensemble = df_combined.groupby(['protein', 'term']).apply(
            lambda g: pd.Series({'score': (g['score'] * g['weight']).sum() / g['weight'].sum()})
        ).reset_index()
    else:
        print("\n[2/4] No ML predictions found, using DIAMOND only...")
        df_ensemble = df_diamond[['protein', 'term', 'score']]
    
    # Save raw ensemble
    ensemble_file = PREDICTIONS_DIR / "ensemble_raw.tsv"
    df_ensemble.to_csv(ensemble_file, sep='\t', index=False, header=False)
    print(f"  Saved raw ensemble: {len(df_ensemble):,} predictions")
    
    # Apply GO propagation
    print("\n[4/4] Applying GO hierarchy propagation...")
    final_file = PREDICTIONS_DIR / "submission_final.tsv"
    
    result = subprocess.run([
        'python3', 'src/propagate_hierarchy.py',
        '--obo', 'Train/go-basic.obo',
        '--infile', str(ensemble_file),
        '--outfile', str(final_file),
        '--min_score', '0.01'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Stats
    df_final = pd.read_csv(final_file, sep='\t', names=['protein', 'term', 'score'])
    
    print(f"\n✅ ENSEMBLE COMPLETE!")
    print(f"   Final predictions: {len(df_final):,}")
    print(f"   Proteins: {df_final['protein'].nunique():,}")
    print(f"   File: {final_file}")
    
    return final_file

if __name__ == "__main__":
    final_file = create_ensemble()
    
    if final_file:
        print("\n" + "="*80)
        print("📁 READY TO SUBMIT!")
        print("="*80)
        print(f"\nFile: {final_file}")
        print("\nUpload to: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/submit")
