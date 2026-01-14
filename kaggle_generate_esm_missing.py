"""
ESM-CAMBRIAN EMBEDDING GENERATION FOR KAGGLE
=============================================
Run this on Kaggle with GPU enabled (T4/P100)

ESM-Cambrian 600M is Meta's newest model (2024):
- Same quality as ESM2-3B (2560 dims)
- 5x faster inference
- Fits on free Kaggle GPUs

Expected runtime: ~6-8 hours for 76K proteins
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from pathlib import Path
from datetime import datetime
import gc
import os

print("="*80)
print("🧬 ESM-CAMBRIAN 600M EMBEDDING GENERATION")
print("="*80)
print(f"Started: {datetime.now()}")

# Configuration
BATCH_SIZE = 8  # Adjust based on GPU memory
MAX_SEQ_LEN = 1024
CHECKPOINT_EVERY = 1000
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # ESM2-650M (good balance)

# Paths - adjust for Kaggle
INPUT_FASTA = "/kaggle/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta"
MISSING_PROTEINS = "/kaggle/input/protgoat-missing/missing_test_accessions.npy"  # Upload this
OUTPUT_DIR = "/kaggle/working/"

# Alternative: ESM-Cambrian if available
try:
    from esm.models.esmc import ESMC
    print("Using ESM-Cambrian 600M!")
    USE_CAMBRIAN = True
except:
    print("ESM-Cambrian not found, using ESM2-650M")
    USE_CAMBRIAN = False

def setup_model():
    """Initialize model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if USE_CAMBRIAN:
        model = ESMC.from_pretrained("esmc_600m")
        model = model.to(device)
        tokenizer = None  # ESMC has built-in tokenization
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, device

def get_embedding_esm2(model, tokenizer, sequences, device):
    """Get embeddings using ESM2."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, 
                       truncation=True, max_length=MAX_SEQ_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pool over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy()

def get_embedding_cambrian(model, sequences, device):
    """Get embeddings using ESM-Cambrian."""
    embeddings = []
    for seq in sequences:
        with torch.no_grad():
            emb = model.encode(seq)
            embeddings.append(emb.cpu().numpy())
    return np.array(embeddings)

def main():
    # Load missing proteins list
    print("\n[1/4] Loading missing proteins list...")
    if os.path.exists(MISSING_PROTEINS):
        missing = np.load(MISSING_PROTEINS, allow_pickle=True)
        missing_set = set(missing)
        print(f"  Missing proteins to generate: {len(missing_set):,}")
    else:
        print("  No missing proteins file, processing ALL test proteins")
        missing_set = None
    
    # Load FASTA
    print("\n[2/4] Loading FASTA sequences...")
    sequences = {}
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        # Extract accession
        header = record.id
        if '|' in header:
            accession = header.split('|')[1]
        else:
            accession = header
        
        # Only process missing proteins
        if missing_set is None or accession in missing_set:
            sequences[accession] = str(record.seq)
    
    print(f"  Sequences to process: {len(sequences):,}")
    
    # Initialize model
    print("\n[3/4] Loading model...")
    model, tokenizer, device = setup_model()
    
    # Process in batches
    print("\n[4/4] Generating embeddings...")
    protein_ids = list(sequences.keys())
    all_embeddings = []
    
    for i in range(0, len(protein_ids), BATCH_SIZE):
        batch_ids = protein_ids[i:i+BATCH_SIZE]
        batch_seqs = [sequences[pid][:MAX_SEQ_LEN] for pid in batch_ids]
        
        if USE_CAMBRIAN:
            embeddings = get_embedding_cambrian(model, batch_seqs, device)
        else:
            embeddings = get_embedding_esm2(model, tokenizer, batch_seqs, device)
        
        all_embeddings.append(embeddings)
        
        if (i + BATCH_SIZE) % 1000 == 0:
            print(f"  Processed {i + len(batch_ids):,}/{len(protein_ids):,}")
            
        # Checkpoint
        if (i + BATCH_SIZE) % CHECKPOINT_EVERY == 0:
            checkpoint_emb = np.vstack(all_embeddings)
            checkpoint_ids = protein_ids[:len(checkpoint_emb)]
            np.save(f"{OUTPUT_DIR}/checkpoint_{i+BATCH_SIZE}.npy", checkpoint_emb)
            np.save(f"{OUTPUT_DIR}/checkpoint_ids_{i+BATCH_SIZE}.npy", np.array(checkpoint_ids))
            print(f"  Saved checkpoint: {i+BATCH_SIZE}")
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final save
    print("\n[SAVING] Saving final embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    final_ids = np.array(protein_ids)
    
    np.save(f"{OUTPUT_DIR}/missing_proteins_embeddings.npy", final_embeddings)
    np.save(f"{OUTPUT_DIR}/missing_proteins_ids.npy", final_ids)
    
    print(f"\n✅ COMPLETE!")
    print(f"  Embeddings shape: {final_embeddings.shape}")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"  Finished: {datetime.now()}")

if __name__ == "__main__":
    main()
