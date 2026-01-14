"""
Generate ESM-Cambrian 600M embeddings for CAFA-6 proteins
Upgrade from ESM2 - better performance, same memory footprint
"""

import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO
from tqdm import tqdm
import os

print("="*80)
print("ESM-Cambrian 600M Embedding Generation for CAFA-6")
print("="*80)

# Configuration
MODEL_NAME = "facebook/esm-cambrian-600M"  # Upgrade from ESM2!
BATCH_SIZE = 4  # Adjust based on GPU memory
MAX_LENGTH = 1024
SAVE_EVERY = 500
OUTPUT_DIR = "/Volumes/TRANSCEND/cafa6_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_embeddings(fasta_file, output_prefix, device="cuda"):
    """Generate ESM-Cambrian embeddings for proteins"""
    
    print(f"\n📁 Input: {fasta_file}")
    print(f"💾 Output: {OUTPUT_DIR}/{output_prefix}_*.npz")
    print(f"🖥️  Device: {device}")
    
    # Load model
    print("\n[1/4] Loading ESM-Cambrian 600M model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        device = "cpu"
        print(f"⚠️  Using CPU (slower)")
    
    model.eval()
    
    # Load sequences
    print("\n[2/4] Loading sequences...")
    records = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"✅ Found {len(records):,} proteins")
    
    # Process in batches
    print(f"\n[3/4] Generating embeddings (batch size={BATCH_SIZE})...")
    
    all_protein_ids = []
    all_embeddings = []
    
    for i in tqdm(range(0, len(records), BATCH_SIZE)):
        batch_records = records[i:i+BATCH_SIZE]
        
        # Prepare batch
        protein_ids = []
        sequences = []
        
        for record in batch_records:
            pid = record.id.split('|')[1] if '|' in record.id else record.id
            seq = str(record.seq)[:MAX_LENGTH]  # Truncate if too long
            
            protein_ids.append(pid)
            sequences.append(seq)
        
        # Tokenize
        inputs = tokenizer(sequences, return_tensors="pt", 
                          padding=True, truncation=True, 
                          max_length=MAX_LENGTH)
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Move to CPU and convert to numpy
        embeddings = embeddings.cpu().numpy()
        
        all_protein_ids.extend(protein_ids)
        all_embeddings.append(embeddings)
        
        # Save checkpoint
        if (i + BATCH_SIZE) % SAVE_EVERY == 0:
            checkpoint_file = f"{OUTPUT_DIR}/{output_prefix}_checkpoint_{i}.npz"
            embeddings_so_far = np.vstack(all_embeddings)
            np.savez(checkpoint_file,
                    protein_ids=np.array(all_protein_ids),
                    embeddings=embeddings_so_far)
            print(f"\n  💾 Checkpoint saved: {checkpoint_file}")
    
    # Final save
    print("\n[4/4] Saving final embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    
    output_file = f"{OUTPUT_DIR}/{output_prefix}_final.npz"
    np.savez(output_file,
            protein_ids=np.array(all_protein_ids),
            embeddings=final_embeddings)
    
    print(f"\n✅ COMPLETE!")
    print(f"   Saved: {output_file}")
    print(f"   Shape: {final_embeddings.shape}")
    print(f"   Embedding dim: {final_embeddings.shape[1]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output prefix")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    
    args = parser.parse_args()
    
    generate_embeddings(args.fasta, args.output, args.device)

# Usage on Kaggle:
# python generate_esm_cambrian.py \
#   --fasta /kaggle/input/cafa-6/Test/testsuperset.fasta \
#   --output test_esm_cambrian \
#   --device cuda
