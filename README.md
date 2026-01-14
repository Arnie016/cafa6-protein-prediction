# CAFA-6 Protein Function Prediction 🧬

> Multi-modal ensemble approach for Gene Ontology term prediction

## 🏆 Competition
[Kaggle CAFA-6](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) - Critical Assessment of Functional Annotation

## 📊 Results
| Method | F-max Score |
|--------|-------------|
| DIAMOND Baseline | 0.219 |
| Text Mining (PubMedBERT) | 0.179 |
| Ensemble v1 | 0.233 |
| Final Ensemble | TBD |

## 🧠 Models Used
- **ESM2-650M/3B** - Protein language model embeddings
- **ProtT5-XL** - T5-based protein encoder
- **PubMedBERT** - Text mining from UniProt descriptions
- **ProstT5** - 3Di structural sequence generation

## 🚀 Quick Start

### Modal Cloud (Recommended)
```bash
# Install Modal
pip install modal
modal setup

# Run structure model
python3 -m modal run --detach modal_structure.py

# Run text mining
python3 -m modal run --detach modal_text_mining.py
```

### Ensemble
```bash
python ensemble_predictions.py \
    --inputs sub1.tsv sub2.tsv \
    --weights 0.5 0.5 \
    --output final.tsv
```

## 📁 Key Files

| File | Description |
|------|-------------|
| `modal_structure.py` | ESM2-650M on Modal A10G |
| `modal_structure_3B.py` | ESM2-3B on Modal A100 |
| `modal_prott5_fixed.py` | ProtT5-XL on Modal A100 |
| `modal_text_mining.py` | PubMedBERT text mining |
| `ensemble_predictions.py` | Weighted ensemble |
| `enforce_hierarchy.py` | GO DAG consistency |
| `LEARNINGS.md` | Pitfalls & lessons learned |

## 📚 Documentation
- [LEARNINGS.md](LEARNINGS.md) - Complete learnings & pitfalls
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Technical details

## ⚠️ Requirements
- Python 3.9+
- PyTorch, Transformers, Biopython
- Modal account (for cloud GPU)

## 📖 License
MIT

---
*CAFA-6 Competition - January 2026*
