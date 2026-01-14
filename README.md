# CAFA-6 Protein Function Prediction 🧬

> Multi-modal ensemble for Gene Ontology term prediction

## 📊 Score Progression

| Stage | Method | F-max | Notes |
|-------|--------|-------|-------|
| 1 | LightGBM Baseline | 0.114 | AAC features only |
| 1 | DIAMOND Homology | **0.219** | Sequence alignment |
| 2 | Ensemble v1 | **0.233** | Combined baseline |
| 3 | PubMedBERT Text Mining | 0.179 | UniProt descriptions |
| 4 | ESM2-650M + MLP | TBD | Modal A10G |
| 4 | ESM2-3B + MLP | TBD | Modal A100 |
| 4 | ProtT5-XL + MLP | TBD | Modal A100 |
| 5 | Final Ensemble | **TBD** | All models combined |

## 📁 Folder Structure

```
├── 01_baseline/           # Starting point (Score: 0.219)
│   └── diamond_baseline.py
│
├── 02_text_mining/        # PubMedBERT (Score: 0.179)
│   └── modal_pubmedbert.py
│
├── 03_structure_models/   # PLM embeddings (Score: TBD)
│   ├── modal_esm2_650M.py
│   ├── modal_esm2_3B.py
│   └── modal_prott5.py
│
├── 04_advanced/           # 3Di structural search
│   ├── generate_3di.py
│   ├── 3di_matrix.mat
│   └── transfer_3di_go.py
│
├── 05_ensemble/           # Final combination
│   ├── ensemble_predictions.py
│   ├── tune_ensemble.py
│   ├── enforce_hierarchy.py
│   └── correct_with_graph.py
│
└── utils/                 # Helper scripts
    ├── fetch_uniprot_descriptions.py
    ├── fetch_string_ppi.py
    └── filter_by_taxonomy.py
```

## 🚀 Quick Start

```bash
# 1. Run baseline
python 01_baseline/diamond_baseline.py

# 2. Run Modal models (cloud GPU)
modal run 03_structure_models/modal_esm2_650M.py

# 3. Ensemble
python 05_ensemble/ensemble_predictions.py --inputs *.tsv
```

## 📚 Key Learnings

See [LEARNINGS.md](LEARNINGS.md) for:
- ⚠️ Pitfalls (Modal crashes, budget limits)
- 💡 Tips (--detach flag, A100 for 3B models)
- 🔧 Technical decisions

## 📖 License
MIT
