# Plan to Achieve CAFA 6 Top Tier Performance (Score ~0.68)

## 1. Setup & Data (Immediate)
- **External Storage**: Configure "Transcend" drive for storing large embeddings.
- **Data Loading**: Ensure all FASTA and Taxonomy data is loaded correctly.
- **Checkpointing**: Completed setup for robust embedding generation.

## 2. Feature Engineering: Protein Embeddings
- **Tool**: ESM-2 (Evolutionary Scale Modeling)
- **Model Size**: Start with `esm2_t6_8M` for debugging, likely need `esm2_t33_650M` (3GB params) or `esm2_t36_3B` for winning scores.
- **Action**: Run `src/generate_embeddings.py` on the full dataset. This is the most time-consuming step (hours to days depending on GPU).
- **Goal**: Convert every protein sequence into a rich vector of fixed size (e.g., 1280 dim).

## 3. Homology Search (The "Alignment Baseline" fix)
- **Tool**: Diamond (BLAST-like aligner).
- **Action**: Fix local installation of Diamond.
- **Logic**: If a test protein is 90% identical to a known training protein, we simply copy the training labels. This is extremely high precision.
- **Integration**: Mix Homology scores with ML scores.

## 4. Advanced Modeling
- **Model**: Multi-label Deep Neural Network (MLP) or Gradient Boosting (XGBoost/LightGBM) on top of ESM embeddings.
- **Taxonomy**: Incorporate Taxon ID (Species) as a feature? (Some functions are species-specific).
- **Ensemble**: Train 5-10 models and average their predictions.

## 5. Evaluation & Submission
- **Metric**: F-max (maximum F1 score across thresholds).
- **Validation**: Create a split (e.g., 80/20) that respects sequence similarity (don't put similar proteins in both train and val).
- **Submission**: Generate correct TSV format.

## Roadmap Steps
1. [x] Naive Baseline
2. [x] Basic AAC Baseline
3. [ ] **Run Full ESM Embeddings** (Waiting on storage config)
4. [ ] **Fix Diamond / Homology Search**
5. [ ] **Train High-Capacity Model**
