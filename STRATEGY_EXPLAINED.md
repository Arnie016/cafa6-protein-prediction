# How We Are Predicting Protein Function (Layman's Guide) 🧬

Imagine you find thousands of mysterious tools in an alien spaceship (these are "Proteins"). You have no idea what they do. Your job is to label them: "This is a wrench," "This generates power," "This breaks down waste."

In biology, this is the **CAFA 6 Challenge** (Critical Assessment of Functional Annotation). We have 224,000 mysterious proteins, and we need to predict their functions (Gene Ontology terms).

Here is our "Master Strategy" to solve this, explained simply.

---

## 1. The "Looking at the Label" Method (Text Mining) 🏷️
**Analogy:** You pick up an alien tool and see a sticker that says "Plasma Cutter 3000". Even if you don't know how it works, the name gives it away.
- **What we do:** Some proteins have descriptions in databases (e.g., "Hemoglobin beta chain").
- **Our AI:** We use a language model called **PubMedBERT** (which has read millions of biology papers) to read these descriptions and guess the function.
- **Why it works:** It's surprisingly accurate for well-studied proteins.

## 2. The "Twin Brother" Method (Sequence Homology) 👯
**Analogy:** You pick up a tool that looks exactly like a hammer you have at home. You assume it's also a hammer.
- **What we do:** We compare the alien protein's amino acid sequence (its DNA recipe) to a database of known proteins.
- **Our Tool:** We use **DIAMOND** (super-fast search) to find "relatives". If Protein A looks 90% like Protein B, they probably do the same thing.
- **Why it works:** Evolution conserves function. If it ain't broke, nature doesn't fix it.

## 3. The "Shape Detective" Method (Structure Models) 🧩
**Analogy:** You find a tool with a handle and a flat heavy head. Even if you've never seen it before, its *shape* tells you it's for hitting things.
- **What we do:** Sometimes two proteins have completely different recipes (sequences) but fold into the exact same 3D shape. Strategies #1 and #2 fail here.
- **Our AI:** We use massive "Protein Language Models" (**ESM2** and **ProtT5**) running on cloud supercomputers (A100 GPUs). These models "understand" the language of proteins and can predict function from abstract patterns, even without a close relative.
- **The Secret Weapon:** We also use **3Di** (a "structural alphabet") to find proteins that *fold* the same way, even if they look different on paper.

## 4. The "Guilt by Association" Method (PPI Networks) 🕸️
**Analogy:** You see a mysterious tool hanging on a belt next to a screwdriver and a wrench. You assume it's also a construction tool, not a kitchen utensil.
- **What we do:** Proteins work in teams. We look at the **Social Network** of proteins (Protein-Protein Interactions).
- **Our Strategy:** If an unknown protein hangs out with 5 "DNA Repair" proteins, it's almost certainly involved in DNA repair.

---

## 5. The "Grand Council" (Ensemble) 🧙‍♂️
Each of the methods above is smart but flawed:
- The **Text** model fails if there's no description.
- The **Twin** method fails if the protein is an "orphan" (no family).
- The **Shape** model is smart but sometimes hallucinates.

**Our Winning Move:**
We don't trust just one. We create a **Council of Experts**.
- Expert 1 (Text) says: "I think it's a kinase (60% sure)."
- Expert 2 (Twin) says: "I've never seen this before."
- Expert 3 (Shape) says: "It looks like a kinase (80% sure)."
- Expert 4 (Network) says: "It hangs out with kinases!"

We combine their votes mathematically. If 3 out of 4 experts agree, we are very confident. This **Ensemble** approach is how competitions are won.

---
*Created for the CAFA 6 Competition*
