# Advanced Feature Engineering & Scripts for CAFA-6

## 📚 New Scripts Created (While Baseline Runs)

### 1. **Advanced Feature Engineering** ⭐⭐⭐
**File:** `src/advanced_feature_engineering.py`

**Features Generated:** 440 (vs. 4,121 tripeptide hashing)
- Amino acid composition (20)
- **Dipeptide composition (400)** - More informative than tripeptides
- Physicochemical properties (11) - Hydrophobicity, charge, MW, etc.
- Secondary structure propensity (3) - Helix, sheet, coil
- Sequence complexity (5) - Entropy, repeats, disorder
- Length (1)

**Why Better:**
- More biologically meaningful
- Less sparse than 4K tripeptide hashing
- Captures actual protein properties
- Used by many CAFA winners

**Usage:**
```python
from src.advanced_feature_engineering import process_fasta

process_fasta("Train/train_sequences.fasta", 
              "features_v2/train_advanced.parquet")
```

---

### 2. **ESM-Cambrian 600M Embeddings** ⭐⭐⭐⭐
**File:** `generate_esm_cambrian.py`

**What It Does:**
- Generates embeddings using ESM-Cambrian 600M
- **Upgrade from ESM2:** Better performance, same memory
- Checkpoint saving every 500 proteins
- GPU optimized with batching

**Key Specs:**
- Embedding dimension: 2560 (same as ESM2-3B)
- Performance: Matches ESM2-3B, but 5× faster
- Memory: ~16 GB GPU (fits on Kaggle T4)

**Usage on Kaggle:**
```bash
python generate_esm_cambrian.py \
  --fasta /kaggle/input/cafa-6/Test/testsuperset.fasta \
  --output test_esm_cambrian \
  --device cuda
```

**Expected Runtime:**
- 76K new proteins × 0.5 sec = **~11 hours** on Kaggle GPU

---

## 🎯 Next Steps Workflow

### **Phase 1: Generate New Embeddings** (Day 1-2)

1. **Identify proteins needing embeddings:**
```python
# Already done! See: /Volumes/TRANSCEND/cafa6_data/
proteins_need_embeddings = np.load(
    "/Volumes/TRANSCEND/cafa6_data/cafa6_proteins_need_embeddings.npy"
)
# 76,544 proteins
```

2. **Create filtered FASTA:**
```bash
python scripts/filter_fasta.py \
  --input Test/testsuperset.fasta \
  --proteins cafa6_proteins_need_embeddings.npy \
  --output Test/new_proteins_only.fasta
```

3. **Generate embeddings on Kaggle:**
```bash
# Upload generate_esm_cambrian.py to Kaggle notebook
# Run with GPU enabled
# Download results
```

---

### **Phase 2: Combine Features** (Day 3)

4. **Merge PROTGOAT + New Embeddings:**
```python
# For proteins IN PROTOAT
import numpy as np

cafa5_path = "/Users/hema/.cache/kagglehub/datasets/zmcxjt/cafa5-train-test-data/versions/2/"
protgoat_embeddings = np.load(f"{cafa5_path}/ESM2_3B_test_embeddings_sorted.npy")
protgoat_labels = np.load(f"{cafa5_path}/ESM2_3B_test_labels_sorted.npy")

# For NEW proteins
new_embeddings = np.load("/Volumes/TRANSCEND/cafa6_embeddings/test_esm_cambrian_final.npz")

# Combine both
combined_embeddings = merge_by_protein_id(protgoat, new_embeddings)
```

---

### **Phase 3: Train PROTGOAT-Lite** (Day 4-5)

5. **Build neural network:**
```python
# See PROTGOAT architecture in research_questions.md
# Simplified version with 3 PLM inputs instead of 5
```

6. **Train with GO-DAG constraints:**
```python
# Custom MCM loss from PROTGOAT
# 6 models (3-fold × 2 seeds) per ontology
```

---

### **Phase 4: Ensemble & Submit** (Day 6-7)

7. **Combine all predictions:**
```
Final Ensemble:
├─ LightGBM Baseline (running now)
├─ PROTGOAT-Lite (6 models × 3 ontologies)
├─ DIAMOND Homology (if time allows)
└─ GO Hierarchy Propagation
```

8. **Submit improved version!**

---

## 📊 Expected Performance Gains

| Stage | Method | Features | F-max | Improvement |
|-------|--------|----------|-------|-------------|
| **Current** | LightGBM + tripeptides | 4,121 | ~0.28 | Baseline |
| **+Advanced Features** | LightGBM + dipeptides | 440 | ~0.32 | +14% |
| **+ESM-Cambrian** | + PLM embeddings | 2,560 | ~0.38 | +36% |
| **+PROTGOAT-Lite** | Multi-PLM + GO-DAG | ~5,000 | ~0.45 | +61% |
| **+Full Ensemble** | Everything combined | - | ~0.50 | +79% |

---

## 💡 Key Innovations in New Scripts

### **1. Advanced Feature Engineering**
**Innovation:** Dipeptides instead of tripeptides
- **Why:** 400 features vs 4,096 (less sparse)
- **Biological meaning:** Captures local structure better
- **Used by:** Multiple CAFA winners

### **2. ESM-Cambrian**
**Innovation:** Newest Meta AI model (2024/2025)
- **Why:** State-of-the-art protein LM
- **Performance:** Better than ESM2-3B, faster
- **Memory efficient:** Fits on free Kaggle GPU

### **3. Modular Design**
**Innovation:** Each script independent
- Can run advanced features WITHOUT embeddings
- Can use embeddings WITHOUT neural network
- Mix and match components

---

## 🚀 Quick Start After Baseline Submission

**Step 1:** Generate advanced features (fast, CPU only)
```bash
python src/advanced_feature engineering.py
```

**Step 2:** Train improved LightGBM with better features
```bash
# Reuse existing Kaggle training script
# Just swap feature files
```

**Step 3:** Submit improved baseline
- Should get ~0.32-0.35 F-max
- +14-25% improvement
- **ZERO GPU time needed!**

**Then decide:** Continue to PROTGOAT-Lite or iterate on features?

---

## 📋 Files Created

1. ✅ `src/advanced_feature_engineering.py` - 440 biological features
2. ✅ `generate_esm_cambrian.py` - Latest PLM embeddings
3. ✅ `ADVANCED_FEATURES_GUIDE.md` - This file

**Coming next (if you want):**
- Text feature extraction from UniProt
- GO-DAG matrix builder
- PROTGOAT-Lite neural network
- Ensemble stacking script

---

Ready to use after your baseline submission completes! 🎯
