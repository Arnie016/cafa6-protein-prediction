# How It Works: The "Council of Experts" Strategy 🧬

**The Challenge:** We have 224,000 proteins. We don't know what they do.
**The Goal:** Predict their function (e.g., "This protein helps repair DNA").

No single method is perfect. Instead, we built a **"Council of Experts"**—four different AI systems that look at the problem from completely different angles. We combine their votes to make the final decision.

---

### 1. The Historian (Sequence Alignment) 📜
*Looks for records of similar proteins.*
- **Logic:** "I've seen a protein almost identical to this one before. It was a DNA repair enzyme. This one probably is too."
- **Tech:** Uses **DIAMOND** (an ultra-fast search tool) to compare our mystery protein against millions of known proteins.
- **Strength:** Extremely accurate when a close relative exists.
- **Weakness:** Useless if the protein is "new" or unique.

### 2. The Linguist (Text Mining) �
*Reads the protein's 'ID card'.*
- **Logic:** "I don't know the structure, but its description says 'Ubiquitin-conjugating enzyme'. I can guess what that does based on the name!"
- **Tech:** Uses **PubMedBERT** (an AI model trained on biological literature) to understand the text descriptions associated with proteins.
- **Strength:** Can solve proteins that have descriptive names but weird structures.

### 3. The Architect (Structural AI) 🏗️
*Analyzes the 3D shape and pattern.*
- **Logic:** "I don't recognize the sequence or the name, but the *shape* of this molecule looks like a key that fits a specific lock."
- **Tech:** Uses **ESM2** and **ProtT5** (Massive AI models similar to ChatGPT, but for protein codes) running on cloud GPUs. We also scan for "3Di" structural patterns.
- **Strength:** The only way to solve "Orphan Proteins" that have no known relatives.

### 4. The Sociologist (PPI Networks) 🕸️
*Checks who the protein hangs out with.*
- **Logic:** "This protein is always found attached to three other 'Metabolism' proteins. It's safe to assume this one is involved in metabolism too."
- **Tech:** Maps the protein to the **STRING Database**, a massive graph of protein-protein interactions.
- **Strength:** Good for confirming predictions using "guilt by association."

---

## 🏆 The Final Verdict (Ensemble)
We don't let any single expert decide.
- If the **Historian** is 90% sure, but the **Architect** says "No way," we treat it with caution.
- If **All 4 Experts** agree, we are confident.

We tune the "voting weight" of each expert mathematically to maximize our score. This multi-view approach is how we aim to win.
