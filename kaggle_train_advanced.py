"""
CAFA-6 Kaggle GPU Training - ADVANCED FEATURES VERSION
========================================================
Upload this to Kaggle with:
1. Your train_advanced.parquet
2. Your test_advanced.parquet  
3. GPU enabled

Expected improvement: 0.30-0.35 F-max (vs 0.11 with tripeptides)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import gc
from datetime import datetime
from sklearn.random_projection import SparseRandomProjection

print("="*80)
print("CAFA-6 GPU Training - Advanced Features (433 biological features)")
print("="*80)

# Configuration
CONFIG = {
    'n_components': 256,        # SRP dimension (smaller than 512 for 433 features)
    'n_terms_per_aspect': 500,  # Top GO terms per aspect
    'n_estimators': 150,        # More trees
    'learning_rate': 0.03,      # Lower for better accuracy
    'num_leaves': 127,          # More leaves
    'max_depth': 10,            # Deeper
    'min_child_samples': 20,
    'reg_alpha': 0.1,           # L1 regularization
    'reg_lambda': 0.1,          # L2 regularization
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
}

# File paths - adjust for your Kaggle setup
# Option 1: Upload as dataset
FEATURE_FILE = "/kaggle/input/cafa6-advanced-features/train_advanced.parquet"
TRAIN_TERMS_FILE = "/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv" 
OUTPUT_DIR = "/kaggle/working/"

# Alternative paths if files are in different location
if not os.path.exists(FEATURE_FILE):
    FEATURE_FILE = "train_advanced.parquet"  # If in working dir
    
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_aspect(aspect, df_features, df_terms, srp):
    """Train models for one GO aspect."""
    print(f"\n{'='*60}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*60}")
    
    # Get top terms for this aspect
    aspect_terms = df_terms[df_terms['aspect'] == aspect]
    term_counts = aspect_terms['term'].value_counts()
    top_terms = term_counts.head(CONFIG['n_terms_per_aspect']).index.tolist()
    print(f"Training {len(top_terms)} GO terms")
    
    # Prepare features
    feat_cols = [c for c in df_features.columns if c not in ['ProteinID', 'TaxonID']]
    X = df_features[feat_cols].values.astype(np.float32)
    
    # Apply SRP
    print("Applying SRP dimensionality reduction...")
    X_reduced = srp.transform(X)
    print(f"Feature dims: {X.shape[1]} -> {X_reduced.shape[1]}")
    
    # Create protein -> terms mapping
    protein_terms = aspect_terms.groupby('EntryID')['term'].apply(set).to_dict()
    protein_ids = df_features['ProteinID'].tolist()
    
    models = {}
    
    for i, term in enumerate(top_terms):
        # Create labels
        y = np.array([1 if term in protein_terms.get(pid, set()) else 0 
                      for pid in protein_ids], dtype=np.int8)
        
        pos = y.sum()
        neg = len(y) - pos
        
        if pos < 10:  # Skip very rare terms
            continue
        
        # Calculate class weight
        scale_pos_weight = neg / pos if pos > 0 else 1
        
        # Train LightGBM
        clf = lgb.LGBMClassifier(
            n_estimators=CONFIG['n_estimators'],
            learning_rate=CONFIG['learning_rate'],
            num_leaves=CONFIG['num_leaves'],
            max_depth=CONFIG['max_depth'],
            min_child_samples=CONFIG['min_child_samples'],
            reg_alpha=CONFIG['reg_alpha'],
            reg_lambda=CONFIG['reg_lambda'],
            scale_pos_weight=scale_pos_weight,
            device=CONFIG['device'],
            gpu_platform_id=CONFIG['gpu_platform_id'],
            gpu_device_id=CONFIG['gpu_device_id'],
            verbose=CONFIG['verbose'],
            n_jobs=1,
        )
        
        clf.fit(X_reduced, y)
        models[term] = clf
        
        if (i + 1) % 50 == 0:
            print(f"  Trained {i+1}/{len(top_terms)} models")
    
    print(f"✅ Trained {len(models)} models for aspect {aspect}")
    return models, top_terms

def main():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    
    # Load features
    df_features = pd.read_parquet(FEATURE_FILE)
    print(f"Loaded {len(df_features):,} proteins with {len(df_features.columns)-1} features")
    
    # Load training terms
    df_terms = pd.read_csv(TRAIN_TERMS_FILE, sep='\t', names=['EntryID', 'term', 'aspect'])
    print(f"Loaded {len(df_terms):,} annotations")
    
    # Initialize SRP
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing SRP...")
    feat_cols = [c for c in df_features.columns if c not in ['ProteinID', 'TaxonID']]
    srp = SparseRandomProjection(n_components=CONFIG['n_components'], random_state=42)
    srp.fit(df_features[feat_cols].values[:1000])  # Fit on sample
    
    # Save SRP
    srp_path = f"{OUTPUT_DIR}/srp_advanced.pkl"
    joblib.dump(srp, srp_path)
    print(f"Saved SRP: {srp_path}")
    
    # Train each aspect
    for aspect in ['F', 'P', 'C']:
        models, terms = train_aspect(aspect, df_features, df_terms, srp)
        
        # Save models
        model_path = f"{OUTPUT_DIR}/lgbm_{aspect}_advanced.pkl"
        joblib.dump({'models': models, 'terms': terms}, model_path)
        print(f"Saved: {model_path}")
        
        gc.collect()
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files created:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.pkl'):
            size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024 / 1024
            print(f"  {f}: {size:.1f} MB")

if __name__ == "__main__":
    main()
