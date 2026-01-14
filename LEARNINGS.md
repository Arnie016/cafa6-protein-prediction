# CAFA-6 Protein Function Prediction: Learnings & Pitfalls

> A comprehensive guide based on our competition experience (January 2026)

## 🏆 Competition Overview
- **Goal:** Predict Gene Ontology (GO) terms for 224,000+ proteins
- **Metric:** F-max (harmonic mean of precision/recall at optimal threshold)
- **Best Baseline:** 0.233 (DIAMOND homology)
- **Target:** 0.50+ (Top team territory)

---

## 📚 Key Learnings

### 1. Multi-Modal Ensembling is King
Single models plateau. Winners combine **orthogonal signals**:
- **Sequence Homology** (DIAMOND/BLAST) - Baseline 0.219
- **Text Mining** (PubMedBERT on UniProt descriptions) - 0.179
- **Structure Embeddings** (ESM2-650M, ESM2-3B, ProtT5-XL)
- **Protein Networks** (STRING PPI)
- **3Di Structural Alignment** (ProstT5 + Foldseek matrix)

### 2. Cloud GPUs are Essential
- **Kaggle:** 12-hour limit, often insufficient
- **Modal:** No time limit, A10G/A100 available, `--detach` mode for reliability
- **Cost:** ~$20-30 for full pipeline (cheaper than losing competition)

### 3. GO Hierarchy Matters
- Gene Ontology is a DAG (Directed Acyclic Graph)
- **Rule:** If you predict a child term, you MUST predict all ancestors
- **Implementation:** `enforce_hierarchy.py` - propagate scores upward

### 4. Taxonomy Filtering Helps
- Not all GO terms apply to all organisms
- Plants don't have "neuron development"
- Filter predictions by organism-appropriate terms

---

## ⚠️ Pitfalls & How to Avoid Them

### Pitfall 1: Modal Jobs Die When Computer Sleeps
**Symptom:** Jobs show "CANCELED" or "STOPPED" after laptop lid closes
**Solution:** Always use `--detach` flag:
```bash
python3 -m modal run --detach script.py
```

### Pitfall 2: Budget Limits Pause Jobs
**Symptom:** Job stuck at "Waiting to be scheduled"
**Solution:** Increase Modal budget limit in dashboard ($100 → $200)

### Pitfall 3: STRING PPI Has Sparse Coverage
**Symptom:** `ppi_network_mapped.tsv` has 0 interactions
**Reality:** STRING doesn't cover all protein families
**Mitigation:** Use PPI as optional signal, not primary

### Pitfall 4: Memory Explosion with Large Embeddings
**Symptom:** GPU OOM errors
**Solutions:**
- Use `torch.cuda.empty_cache()` between batches
- Process in smaller batches (8-16 for 3B models)
- Use `float16` instead of `float32`

### Pitfall 5: 3Di Requires Custom Substitution Matrix
**Symptom:** BLOSUM62 gives meaningless alignments for 3Di
**Solution:** Use Foldseek's 3Di matrix (`3di_matrix.mat`)

### Pitfall 6: Model Checkpoint Loss
**Symptom:** Job restarts from 0% after crash
**Solution:** Save intermediate embeddings to disk (not just RAM)

---

## 🛠️ Technical Stack

### Models Used
| Model | Parameters | Use Case |
|-------|------------|----------|
| ESM2-650M | 650M | Fast structure embeddings |
| ESM2-3B | 3B | High-quality structure |
| ProtT5-XL | 3B | Google's protein T5 |
| PubMedBERT | 110M | Text mining |
| ProstT5 | 3B | AA → 3Di translation |

### Key Libraries
```
torch, transformers, biopython, obonet, pandas, numpy
scikit-learn, networkx, modal, tqdm
```

### Cloud Setup (Modal)
```python
image = modal.Image.debian_slim()
    .pip_install("torch", "transformers", ...)
    .add_local_file("data.fasta", "/root/data.fasta")

@app.function(gpu="A100", timeout=60*60*12)
def train(): ...
```

---

## 📁 Key Scripts

| Script | Purpose |
|--------|---------|
| `modal_structure.py` | ESM2-650M on A10G |
| `modal_structure_3B.py` | ESM2-3B on A100 |
| `modal_prott5_fixed.py` | ProtT5-XL on A100 |
| `modal_text_mining.py` | PubMedBERT |
| `generate_3di.py` | ProstT5 3Di generation |
| `ensemble_predictions.py` | Weighted averaging |
| `tune_ensemble.py` | Weight optimization |
| `enforce_hierarchy.py` | GO DAG consistency |
| `filter_by_taxonomy.py` | Organism filtering |
| `correct_with_graph.py` | Graph diffusion |

---

## 📈 Score Progression

| Date | Method | F-max |
|------|--------|-------|
| Day 1 | LightGBM baseline | 0.114 |
| Day 2 | DIAMOND homology | 0.219 |
| Day 2 | Ensemble v1 | 0.233 |
| Day 3 | PubMedBERT (Modal) | 0.179 |
| Day 4 | Multi-model ensemble | TBD |

---

## 🚀 Future Improvements

1. **AlphaFold pLDDT Weighting:** Use structure confidence as feature
2. **Contrastive Learning:** Train GO embeddings with protein alignments
3. **Label Propagation:** GNN over protein similarity graph
4. **FoldSeek Direct:** If PDB files available, use actual 3D alignment

---

## 💡 Pro Tips

1. **Always `--detach` for long Modal jobs**
2. **Save embeddings to disk, not just RAM**
3. **Use A100 for 3B+ parameter models**
4. **Ensemble early, ensemble often**
5. **Test on validation split before Kaggle submit**
6. **Keep Amphetamine running on Mac to prevent sleep**

---

## 📖 References

- [CAFA Challenge](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [ESM2 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)
- [ProtT5 Paper](https://ieeexplore.ieee.org/document/9477085)
- [Foldseek Paper](https://www.nature.com/articles/s41587-023-01773-0)
- [DeepGOZero](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i238/6617520)

---

*Generated: January 2026*
*Competition: CAFA-6 Protein Function Prediction*
