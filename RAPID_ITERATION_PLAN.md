# CAFA 6 Rapid Iteration Plan
## Goal: Reach 0.68+ F-max Score

### Phase 1: Foundation (DONE ✓)
- [x] Naive baseline (0.026 initial)
- [x] Alignment baseline (Diamond)
- [x] Combined baseline (expected 0.35-0.45)

### Phase 2: Protein Language Model Embeddings (IN PROGRESS)
**Timeline: 1-2 hours**
- [ ] Generate ESM-2 embeddings for all proteins
  - Train: 82k proteins
  - Test: 224k proteins
- [ ] Train simple classifier on embeddings
- **Expected Score**: 0.45-0.55

### Phase 3: Quick Wins (Next 2-4 hours)
**A. Better Model Architecture**
- [ ] Multi-label neural network (PyTorch/TensorFlow)
- [ ] Separate models per GO aspect (MF, BP, CC)
- [ ] Expected boost: +0.05-0.10

**B. Ensemble Methods**
- [ ] Combine alignment + ESM predictions (weighted average)
- [ ] Train multiple models with different seeds
- [ ] Expected boost: +0.03-0.05

**C. Threshold Optimization**
- [ ] Grid search optimal confidence thresholds per aspect
- [ ] Expected boost: +0.02-0.03

### Phase 4: Advanced Features (4-8 hours)
**A. Larger ESM Model**
- [ ] Switch from esm2_t6_8M to esm2_t33_650M
- [ ] Expected boost: +0.05-0.10

**B. Feature Engineering**
- [ ] Add taxonomy features (species info)
- [ ] Add sequence length, composition features
- [ ] Expected boost: +0.02-0.04

**C. GO Hierarchy**
- [ ] Use GO ontology structure (parent-child relationships)
- [ ] Propagate predictions up/down the hierarchy
- [ ] Expected boost: +0.03-0.05

### Phase 5: Competition-Specific Tricks (2-4 hours)
**A. Data Augmentation**
- [ ] Use IA.tsv (Information Accretion) if available
- [ ] Cross-validation with proper sequence similarity splits

**B. Post-processing**
- [ ] Remove inconsistent predictions (e.g., child without parent)
- [ ] Calibrate probabilities per GO term

### Phase 6: Final Push (1-2 hours)
- [ ] Ensemble 5-10 best models
- [ ] Fine-tune thresholds on validation set
- [ ] Submit final predictions

---

## Execution Strategy for Rapid Iteration

### Parallel Workstreams
1. **GPU Track**: ESM embeddings → Neural networks
2. **CPU Track**: Feature engineering, ensembling
3. **Analysis Track**: Error analysis, validation

### Quick Feedback Loop
1. Train model (10-30 min)
2. Generate submission (5-10 min)
3. Submit to competition (instant)
4. Get score (instant)
5. Analyze errors (5-10 min)
6. Iterate

### Priority Order (by ROI)
1. **ESM Embeddings** - Highest impact
2. **Ensemble alignment + ESM** - Quick win
3. **Threshold optimization** - Easy boost
4. **Larger ESM model** - If time permits
5. **GO hierarchy** - Advanced but powerful

---

## Current Status
- ✅ Alignment baseline ready
- ⏳ ESM embeddings: Restarting now
- 📊 Expected timeline to 0.60+: 6-12 hours
- 🎯 Expected timeline to 0.68+: 12-24 hours
