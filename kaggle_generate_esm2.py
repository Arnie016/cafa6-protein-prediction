"""
CAFA-6 ESM-2 Embedding Generation for Kaggle GPU
Upload to Kaggle Notebook with GPU T4 x2 enabled

This script generates ESM-2 650M embeddings for train/test sequences.
Estimated runtime: 4-6 hours on Kaggle GPU
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from datetime import datetime
import gc

print("=" * 80)
print("CAFA-6 ESM-2 650M Embedding Generation")
print("=" * 80)

# Configuration
CONFIG = {
    'model_name': 'facebook/esm2_t33_650M_UR50D',  # ESM-2 650M
    'batch_size': 4,  # Small batches for 650M model
    'max_length': 1024,  # Max protein length
    'embedding_dim': 1280,  # ESM-2 650M output dimension
    'save_every': 1000,  # Save checkpoint every N sequences
}

# Kaggle paths
TRAIN_FASTA = "/kaggle/input/cafa-6-protein-function-prediction/Train/train_sequences.fasta"
TEST_FASTA = "/kaggle/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta"
OUTPUT_DIR = "/kaggle/working/"

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def extract_protein_id(record):
    """Extract clean protein ID from FASTA record."""
    # Handle formats like "A0A0C5B5G6 9606" or "sp|A0A0C5B5G6|NAME"
    raw_id = record.id
    if "|" in raw_id:
        return raw_id.split("|")[1]
    return raw_id.split()[0]

def generate_embeddings_batch(sequences, model, tokenizer, device):
    """Generate embeddings for a batch of sequences."""
    # Tokenize
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CONFIG['max_length']
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over sequence length
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Expand attention mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_hidden / sum_mask
    
    return embeddings.cpu().numpy()

def save_checkpoint(output_path, protein_ids, embeddings, end_idx):
    """Save embeddings checkpoint."""
    checkpoint_path = output_path.replace(".npz", f"_checkpoint_{end_idx}.npz")
    np.savez_compressed(
        checkpoint_path,
        ids=np.array(protein_ids),
        embeddings=embeddings
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint: {checkpoint_path}")
    return checkpoint_path

def process_fasta(fasta_path, output_path, model, tokenizer, device, prefix="train"):
    """Process FASTA file and generate embeddings."""
    print(f"\n{'='*80}")
    print(f"Processing {prefix.upper()} sequences from: {fasta_path}")
    print(f"{'='*80}")
    
    # Count total sequences first
    total_seqs = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    print(f"Total sequences: {total_seqs:,}\n")
    
    all_ids = []
    all_embeddings = []
    batch_ids = []
    batch_seqs = []
    
    checkpoint_files = []
    processed = 0
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        protein_id = extract_protein_id(record)
        sequence = str(record.seq)[:CONFIG['max_length']]
        
        batch_ids.append(protein_id)
        batch_seqs.append(sequence)
        
        # Process batch when full
        if len(batch_ids) == CONFIG['batch_size']:
            embeddings = generate_embeddings_batch(batch_seqs, model, tokenizer, device)
            all_embeddings.append(embeddings)
            all_ids.extend(batch_ids)
            processed += len(batch_ids)
            
            # Progress logging
            if processed % 100 == 0:
                pct = (processed / total_seqs) * 100
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {processed:,}/{total_seqs:,} ({pct:.1f}%)")
            
            # Save checkpoint periodically
            if len(all_ids) >= CONFIG['save_every']:
                embeddings_array = np.vstack(all_embeddings)
                ckpt_path = save_checkpoint(output_path, all_ids, embeddings_array, processed)
                checkpoint_files.append(ckpt_path)
                
                # Clear memory
                all_ids = []
                all_embeddings = []
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Reset batch
            batch_ids = []
            batch_seqs = []
    
    # Process remaining sequences
    if batch_ids:
        embeddings = generate_embeddings_batch(batch_seqs, model, tokenizer, device)
        all_embeddings.append(embeddings)
        all_ids.extend(batch_ids)
        processed += len(batch_ids)
    
    # Save final checkpoint
    if all_ids:
        embeddings_array = np.vstack(all_embeddings)
        ckpt_path = save_checkpoint(output_path, all_ids, embeddings_array, processed)
        checkpoint_files.append(ckpt_path)
    
    print(f"\n✅ Completed {prefix}: {processed:,} sequences")
    print(f"Checkpoint files: {len(checkpoint_files)}")
    
    return checkpoint_files

def merge_checkpoints(checkpoint_files, output_path):
    """Merge all checkpoint files into final embedding file."""
    print(f"\nMerging {len(checkpoint_files)} checkpoints...")
    
    all_ids = []
    all_embeddings = []
    
    for ckpt_file in sorted(checkpoint_files):
        data = np.load(ckpt_file)
        all_ids.extend(data['ids'])
        all_embeddings.append(data['embeddings'])
        print(f"  Loaded: {ckpt_file}")
    
    # Combine
    final_embeddings = np.vstack(all_embeddings)
    final_ids = np.array(all_ids)
    
    # Save final file
    np.savez_compressed(
        output_path,
        ids=final_ids,
        embeddings=final_embeddings
    )
    
    print(f"\n✅ Saved final embeddings: {output_path}")
    print(f"   Shape: {final_embeddings.shape}")
    print(f"   Size: {final_embeddings.nbytes / 1024**2:.1f} MB")
    
    return output_path

def main():
    # Initialize device
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"\nLoading ESM-2 650M model...")
    print("(This may take 2-3 minutes on first run)")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModel.from_pretrained(CONFIG['model_name'])
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    print(f"   Parameters: ~650M")
    print(f"   Embedding dimension: {CONFIG['embedding_dim']}")
    
    # Process train sequences
    train_output = os.path.join(OUTPUT_DIR, "esm2_train_embeddings.npz")
    train_checkpoints = process_fasta(
        TRAIN_FASTA,
        train_output,
        model,
        tokenizer,
        device,
        prefix="train"
    )
    merge_checkpoints(train_checkpoints, train_output)
    
    # Clear memory before test
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Process test sequences
    test_output = os.path.join(OUTPUT_DIR, "esm2_test_embeddings.npz")
    test_checkpoints = process_fasta(
        TEST_FASTA,
        test_output,
        model,
        tokenizer,
        device,
        prefix="test"
    )
    merge_checkpoints(test_checkpoints, test_output)
    
    print("\n" + "=" * 80)
    print("✅ ALL EMBEDDINGS GENERATED!")
    print("=" * 80)
    print("\nDownload these files from Kaggle:")
    print("  - esm2_train_embeddings.npz")
    print("  - esm2_test_embeddings.npz")
    print("\nNext steps:")
    print("  1. Download embeddings to local machine")
    print("  2. Merge with basic features using merge_features script")
    print("  3. Upload merged features to Kaggle for training")

if __name__ == "__main__":
    main()
