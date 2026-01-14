"""
CAFA-6 GPU Training Script for Kaggle
Upload this to a Kaggle Notebook with GPU enabled
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import gc
from datetime import datetime
from sklearn.random_projection import SparseRandomProjection
import pyarrow.parquet as pq

print("=" * 80)
print("CAFA-6 GPU Training - LightGBM with SRP")
print("=" * 80)

# Configuration
CONFIG = {
    'n_components': 512,        # SRP dimension reduction
    'n_terms_per_aspect': 500,  # Increased from 200 to 500!
    'n_estimators': 100,        # Increased from 20 to 100!
    'learning_rate': 0.05,      # Lower for better accuracy
    'num_leaves': 63,           # Increased from 15
    'max_depth': 8,             # Increased from 3
    'device': 'gpu',            # Use GPU!
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

def train_lgbm_gpu():
    # Paths (adjust for Kaggle environment)
    # In Kaggle, your data will be in /kaggle/input/
    FEATURE_FILE = "/kaggle/input/cafa-6-features/train_features.parquet"
    TRAIN_TERMS_FILE = "/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    OUTPUT_DIR = "/kaggle/working/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. SETUP SRP
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing SRP...")
    df_head = pd.read_parquet(FEATURE_FILE).head(1)
    feat_cols = [c for c in df_head.columns if c not in ['ProteinID', 'TaxonID']]
    print(f"Feature count: {len(feat_cols)}")
    
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
    train_ids = np.array(ids_list)
    
    print(f"Final shape: {X_final.shape}")
    print(f"Memory: {X_final.nbytes / 1024**2:.2f} MB")
    
    del X_reduced_list
    gc.collect()
    
    # 3. LOAD LABELS
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading labels...")
    train_terms = pd.read_csv(TRAIN_TERMS_FILE, sep='\t')
    
    # 4. TRAIN MODELS
    aspects = ['F', 'P', 'C']
    for aspect in aspects:
        print(f"\n{'='*80}")
        print(f"ASPECT: {aspect}")
        print(f"{'='*80}")
        
        asp_df = train_terms[train_terms['aspect'] == aspect]
        top_terms = asp_df['term'].value_counts().head(CONFIG['n_terms_per_aspect']).index.tolist()
        
        print(f"Training {len(top_terms)} terms for aspect {aspect}")
        
        # Build y_train
        y_train = np.zeros((len(train_ids), len(top_terms)), dtype=np.int8)
        id_map = {pid: i for i, pid in enumerate(train_ids)}
        term_map = {term: i for i, term in enumerate(top_terms)}
        
        filtered = asp_df[asp_df['term'].isin(top_terms)]
        filtered = filtered[filtered['EntryID'].isin(id_map.keys())]
        
        for _, row in filtered.iterrows():
            y_train[id_map[row['EntryID']], term_map[row['term']]] = 1
        
        # Train models
        models_dict = {}
        for i, term in enumerate(top_terms):
            if i % 25 == 0:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Term {i}/{len(top_terms)}: {term}")
            
            y_col = y_train[:, i]
            if y_col.sum() == 0:
                continue
            
            clf = lgb.LGBMClassifier(
                n_estimators=CONFIG['n_estimators'],
                learning_rate=CONFIG['learning_rate'],
                num_leaves=CONFIG['num_leaves'],
                max_depth=CONFIG['max_depth'],
                device=CONFIG['device'],
                gpu_platform_id=CONFIG['gpu_platform_id'],
                gpu_device_id=CONFIG['gpu_device_id'],
                n_jobs=4,
                verbosity=-1
            )
            clf.fit(X_final, y_col)
            models_dict[term] = clf
        
        # Save
        output_file = f"{OUTPUT_DIR}/lgbm_{aspect}_gpu.pkl"
        joblib.dump({'models': models_dict, 'terms': top_terms}, output_file)
        print(f"\n✅ Saved {aspect} model: {output_file}")
        
        del y_train, models_dict
        gc.collect()
    
    print("\n" + "="*80)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*80)
    print("\nDownload these files from Kaggle:")
    print("  - srp_projection.pkl")
    print("  - lgbm_F_gpu.pkl")
    print("  - lgbm_P_gpu.pkl")
    print("  - lgbm_C_gpu.pkl")

if __name__ == "__main__":
    train_lgbm_gpu()
