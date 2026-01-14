# Research Agent Instruction Prompt: CAFA-6 Optimization

## 🎯 Mission
You are a research assistant helping optimize a competitive protein function prediction pipeline for the CAFA-6 (Critical Assessment of Functional Annotation) competition. Your goal is to identify actionable improvements that can boost our F-max score from 0.233 to 0.45-0.50+.

---

## 📋 Current Context

### What We're Doing
Predicting Gene Ontology (GO) terms for 224,000 proteins using a multi-model ensemble approach.

### Our Current Pipeline
We have implemented a **"Council of Experts"** strategy combining:

1. **Sequence Homology (DIAMOND)**
   - Baseline: 0.219 F-max
   - Fast but fails on orphan proteins

2. **Text Mining (PubMedBERT)**
   - Score: 0.179 F-max
   - Uses UniProt descriptions
   - Fails when descriptions are sparse

3. **Structure Models (PLMs on Modal Cloud)**
   - ESM2-650M (A10G GPU) - Currently running
   - ESM2-3B (A100 GPU) - Currently running
   - ProtT5-XL (A100 GPU) - Currently running
   - Expected: 0.35-0.45 F-max per model
   
4. **3Di Structural Search (ProstT5)**
   - Translates AA sequences → 3D structural alphabet
   - Uses Foldseek matrix for alignment
   - Status: Implementation complete, debugging

5. **Ensemble Strategy**
   - Weighted averaging (tune_ensemble.py)
   - Hierarchical consistency (GO DAG enforcement)
   - Graph-based correction

### What's Working
- Cloud GPU pipeline (Modal) running successfully
- All baseline models completed
- Ensemble infrastructure ready

### What's Blocked/Pending
- Waiting for structure models to finish (ETA: 9 AM)
- Need to optimize ensemble weights
- Need to tune prediction thresholds

---

## 🎯 Your Research Task

### Primary Question
**"What are the top 3 techniques used by CAFA winners (2021-2024) that we are NOT currently using, and how can we implement them in the next 24-48 hours?"**

### Specific Areas to Investigate

#### 1. **Score Calibration**
- **Question:** How do top teams calibrate confidence scores before ensembling?
- **Look for:** Platt scaling, isotonic regression, temperature scaling
- **Output:** Python code snippet or algorithm description
- **Constraint:** Must work with pre-trained models (no retraining)

#### 2. **Threshold Optimization**
- **Question:** Should we use a single threshold (0.5) or per-term optimized thresholds?
- **Look for:** F-max optimization strategies, precision-recall curves
- **Output:** Method to find optimal threshold per GO term
- **Metric:** CAFA uses F-max = 2PR/(P+R) at the optimal threshold

#### 3. **Negative Evidence & Open World**
- **Question:** How to handle "absence of evidence" vs "evidence of absence"?
- **Look for:** How winners treat low scores (unknown vs definitely not)
- **Output:** Decision rule or confidence adjustment formula

#### 4. **Taxonomic Filtering**
- **Question:** Are we correctly filtering invalid GO terms by species?
- **Look for:** Best practices for organism-specific GO filtering
- **Output:** Validation that our `filter_by_taxonomy.py` is correct

#### 5. **Error Analysis Frameworks**
- **Question:** Once models finish, how should we identify systematic weaknesses?
- **Look for:** Multi-label classification error analysis methods
- **Output:** A checklist or procedure to run on our predictions

---

## 🔍 How to Help Us

### What We Need
1. **Concrete Techniques:** Not theory—actual implementable methods
2. **Code Examples:** Python snippets or pseudocode
3. **References:** Papers/repos from CAFA 1-5 winners
4. **Priority Ranking:** Which improvements give the biggest boost

### What We DON'T Need
- Suggestions to retrain models from scratch (no time)
- Suggestions requiring new data (we have Train/Test fixed)
- Vague advice like "try deep learning" (we already are)

### Helpful Output Format
```markdown
## Technique: [Name]
- **Impact:** Expected F-max gain (+0.05, +0.10, etc.)
- **Effort:** Hours to implement (2h, 1 day, etc.)
- **Source:** [Paper/GitHub link]
- **Implementation:**
  ```python
  # Pseudo-code here
  ```
- **Integration Point:** Where in our pipeline (before ensemble, after ensemble, etc.)
```

---

## 📊 Key Constraints

- **Time:** Competition deadline approaching
- **Compute:** $200 Modal budget remaining
- **Data:** Cannot modify Train/Test sets
- **Models:** Structure models finishing in 6 hours, cannot restart

---

## 🧠 Meta-Guidance

### Good Research Paths
- Search: "CAFA competition winners techniques"
- Search: "protein function prediction ensemble"
- Search: "multi-label classification calibration"
- Check: Recent BioRxiv/Nature papers (2024-2025)
- Check: GitHub repos with "CAFA" in the name

### Red Flags to Avoid
- Techniques requiring AlphaFold structures (too expensive)
- Techniques requiring PPI networks (we tried, data sparse)
- Techniques requiring labeled validation set (we don't have clean labels)

---

## 🎯 Success Criteria

Your research is successful if:
1. You identify 3 techniques we can implement in <24 hours
2. At least ONE technique is expected to boost F-max by +0.03
3. You provide enough detail that we can code it immediately

---

**Start your research now. Focus on actionable, proven techniques from recent CAFA competitions.**
