"""
CAFA-6 IMPROVED GPU Training Script for Kaggle
Version 2.0 with ESM-2 embeddings, validation, and class balancing

Upload this to a Kaggle Notebook with GPU enabled
Requires: train_features_with_esm.parquet, train_split_ids.txt, val_split_ids.txt
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import gc
from datetime import datetime
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import f1_score
import pyarrow.parquet as pq

print("=" * 80)
print("CAFA-6 GPU Training v2.0 - IMPROVED")
print("ESM-2 Embeddings + Validation + Class Balancing")
print("=" * 80)

# Configuration
CONFIG = {
    'n_components': 512,        # SRP dimension reduction
    'n_terms_per_aspect': 500,  # Top terms to train
    'n_estimators': 300,        # Increased for better performance
    'learning_rate': 0.03,      # Tuned learning rate
    'num_leaves': 63,
    'max_depth': 10,            # Deeper trees
    'min_child_samples': 20,    # Prevent overfitting
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'is_unbalance': True,       # Handle class imbalance
    'early_stopping_rounds': 10,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

def load_validation_split():
    """Load train/val protein IDs."""
    train_ids_path = "/kaggle/input/cafa-6-validation-split/train_split_ids.txt"
    val_ids_path = "/kaggle/input/cafa-6-validation-split/val_split_ids.txt"
    
    if not os.path.exists(train_ids_path) or not os.path.exists(val_ids_path):
        print("⚠️  Validation split files not found, using all data for training")
        return None, None
    
    with open(train_ids_path, 'r') as f:
        train_ids = set([line.strip() for line in f])
    
    with open(val_ids_path, 'r') as f:
        val_ids = set([line.strip() for line in f])
    
    print(f"\n✅ Loaded validation split:")
    print(f"   Train IDs: {len(train_ids):,}")
    print(f"   Val IDs: {len(val_ids):,}")
    
    return train_ids, val_ids

def train_lgbm_gpu():
    # Paths
    FEATURE_FILE = "/kaggle/input/cafa-6-features-esm/train_features_with_esm.parquet"
    TRAIN_TERMS_FILE = "/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    OUTPUT_DIR = "/kaggle/working/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load validation split
    train_split_ids, val_split_ids = load_validation_split()
    use_validation = (train_split_ids is not None)
    
    # 1. SETUP SRP
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing SRP...")
    df_head = pd.read_parquet(FEATURE_FILE).head(1)
    feat_cols = [c for c in df_head.columns if c not in ['ProteinID', 'TaxonID']]
    print(f"Feature count: {len(feat_cols)} (includes ESM-2 embeddings!)")
    
    srp = SparseRandomProjection(
        n_components=CONFIG['n_components'],
        dense_output=True,
        random_state=42
    )
    srp.fit(np.zeros((1, len(feat_cols))))
    joblib.dump(srp, f"{OUTPUT_DIR}/srp_projection.pkl")
    
    # 2. TRANSFORM DATA
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Transforming data...")
    X_reduced_list = []
    ids_list = []
    
    parquet_file = pq.ParquetFile(FEATURE_FILE)
    BATCH_SIZE = 10000
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
        df_batch = batch.to_pandas()
        X_batch = df_batch[feat_cols].values.astype(np.float32)
        X_red_batch = srp.transform(X_batch)
        X_reduced_list.append(X_red_batch)
        ids_list.extend(df_batch['ProteinID'].tolist())
        
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1} processed")
        
        del df_batch, X_batch
        gc.collect()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stacking...")
    X_final = np.vstack(X_reduced_list)
    all_ids = np.array(ids_list)
    
    print(f"Final shape: {X_final.shape}")
    print(f"Memory: {X_final.nbytes / 1024**2:.2f} MB")
    
    del X_reduced_list
    gc.collect()
    
    # 3. CREATE TRAIN/VAL INDICES
    if use_validation:
        train_mask = np.array([pid in train_split_ids for pid in all_ids])
        val_mask = np.array([pid in val_split_ids for pid in all_ids])
        
        X_train = X_final[train_mask]
        X_val = X_final[val_mask]
        train_ids = all_ids[train_mask]
        val_ids = all_ids[val_mask]
        
        print(f"\nSplit created:")
        print(f"  Train: {len(train_ids):,} proteins")
        print(f"  Val: {len(val_ids):,} proteins")
    else:
        X_train = X_final
        X_val = None
        train_ids = all_ids
        val_ids = None
    
    # 4. LOAD LABELS
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading labels...")
    train_terms = pd.read_csv(TRAIN_TERMS_FILE, sep='\t')
    
    # 5. TRAIN MODELS
    aspects = ['F', 'P', 'C']
    for aspect in aspects:
        print(f"\n{'='*80}")
        print(f"ASPECT: {aspect}")
        print(f"{'='*80}")
        
        asp_df = train_terms[train_terms['aspect'] == aspect]
        top_terms = asp_df['term'].value_counts().head(CONFIG['n_terms_per_aspect']).index.tolist()
        
        print(f"Training {len(top_terms)} terms for aspect {aspect}")
        
        # Build y_train
        y_train_full = np.zeros((len(train_ids), len(top_terms)), dtype=np.int8)
        id_map = {pid: i for i, pid in enumerate(train_ids)}
        term_map = {term: i for i, term in enumerate(top_terms)}
        
        filtered = asp_df[asp_df['term'].isin(top_terms)]
        filtered = filtered[filtered['EntryID'].isin(id_map.keys())]
        
        for _, row in filtered.iterrows():
            y_train_full[id_map[row['EntryID']], term_map[row['term']]] = 1
        
        # Build y_val if using validation
        y_val_full = None
        if use_validation:
            y_val_full = np.zeros((len(val_ids), len(top_terms)), dtype=np.int8)
            val_id_map = {pid: i for i, pid in enumerate(val_ids)}
            
            filtered_val = asp_df[asp_df['term'].isin(top_terms)]
            filtered_val = filtered_val[filtered_val['EntryID'].isin(val_id_map.keys())]
            
            for _, row in filtered_val.iterrows():
                y_val_full[val_id_map[row['EntryID']], term_map[row['term']]] = 1
        
        # Train models
        models_dict = {}
        val_scores = {}
        
        for i, term in enumerate(top_terms):
            if i % 25 == 0:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Term {i}/{len(top_terms)}: {term}")
            
            y_train_col = y_train_full[:, i]
            n_pos = y_train_col.sum()
            
            if n_pos == 0:
                continue
            
            # Calculate class weight
            n_neg = len(y_train_col) - n_pos
            scale_pos_weight = n_neg / (n_pos + 1e-9)
            
            # Build model with class balancing
            clf = lgb.LGBMClassifier(
                n_estimators=CONFIG['n_estimators'],
                learning_rate=CONFIG['learning_rate'],
                num_leaves=CONFIG['num_leaves'],
                max_depth=CONFIG['max_depth'],
                min_child_samples=CONFIG['min_child_samples'],
                is_unbalance=CONFIG['is_unbalance'],
                scale_pos_weight=scale_pos_weight if scale_pos_weight < 100 else 100,  # Cap weight
                device=CONFIG['device'],
                gpu_platform_id=CONFIG['gpu_platform_id'],
                gpu_device_id=CONFIG['gpu_device_id'],
                n_jobs=4,
                verbosity=-1
            )
            
            # Train with early stopping if validation available
            if use_validation and y_val_full is not None:
                y_val_col = y_val_full[:, i]
                
                if y_val_col.sum() > 0:  # Only use early stopping if val has positives
                    clf.fit(
                        X_train, y_train_col,
                        eval_set=[(X_val, y_val_col)],
                        eval_metric='binary_logloss',
                        callbacks=[lgb.early_stopping(CONFIG['early_stopping_rounds'], verbose=False)]
                    )
                    
                    # Calculate val F1
                    y_val_pred = (clf.predict_proba(X_val)[:, 1] > 0.5).astype(int)
                    if y_val_pred.sum() > 0 and y_val_col.sum() > 0:
                        val_f1 = f1_score(y_val_col, y_val_pred)
                        val_scores[term] = val_f1
                else:
                    clf.fit(X_train, y_train_col)
            else:
                clf.fit(X_train, y_train_col)
            
            models_dict[term] = clf
        
        # Save
        output_file = f"{OUTPUT_DIR}/lgbm_{aspect}_gpu_v2.pkl"
        joblib.dump({'models': models_dict, 'terms': top_terms, 'val_scores': val_scores}, output_file)
        
        # Report validation performance
        if val_scores:
            mean_f1 = np.mean(list(val_scores.values()))
            print(f"\n📊 Validation F1 (mean): {mean_f1:.4f}")
            print(f"   Terms evaluated: {len(val_scores)}")
        
        print(f"\n✅ Saved {aspect} model: {output_file}")
        
        del y_train_full, y_val_full, models_dict
        gc.collect()
    
    print("\n" + "="*80)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*80)
    print("\nDownload these files from Kaggle:")
    print("  - srp_projection.pkl")
    print("  - lgbm_F_gpu_v2.pkl")
    print("  - lgbm_P_gpu_v2.pkl")
    print("  - lgbm_C_gpu_v2.pkl")

if __name__ == "__main__":
    train_lgbm_gpu()
