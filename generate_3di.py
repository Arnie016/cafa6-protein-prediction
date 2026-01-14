"""
Generate 3Di structural sequences using Rostlab/ProstT5.
Converts Amino Acid sequences to Foldseek's 3Di alphabet.

Usage (Modal for GPU):
    modal run generate_3di.py

Output:
    train_3di.fasta
    test_3di.fasta
"""

import modal

app = modal.App("cafa-6-prostt5")

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "biopython",
        "tqdm"
    )
    .add_local_file("Train/train_sequences.fasta", "/root/data/Train/train_sequences.fasta")
    .add_local_file("Test/testsuperset.fasta", "/root/data/Test/testsuperset.fasta")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,  # 12 hours
    retries=0
)
def generate_3di_sequences():
    import torch
    from transformers import T5Tokenizer, T5EncoderModel
    from Bio import SeqIO
    from tqdm import tqdm
    import re
    
    print("=" * 80)
    print("🧬 ProstT5: Generating 3Di Structural Sequences")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load ProstT5 model
    print("\n[1/4] Loading ProstT5 model...")
    model_name = "Rostlab/ProstT5"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    
    def seq_to_3di(sequences, batch_size=8):
        """Convert amino acid sequences to 3Di using ProstT5."""
        results = []
        
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i+batch_size]
            
            # ProstT5 expects space-separated amino acids
            # and uses a special format: "<AA2fold>" prefix for AA->3Di
            batch_formatted = ["<AA2fold>" + " ".join(list(seq)) for seq in batch]
            
            with torch.no_grad():
                inputs = tokenizer.batch_encode_plus(
                    batch_formatted,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)
                
                # Generate 3Di tokens
                # ProstT5 is an encoder-decoder; we use the decoder to generate
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max(len(s) for s in batch) + 10,
                    num_beams=1,
                    do_sample=False
                )
                
                # Decode back to strings
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Clean up: remove spaces, convert to uppercase
                for dec in decoded:
                    # 3Di uses lowercase a-y, convert to uppercase A-Y for DIAMOND
                    clean = re.sub(r'\s+', '', dec).upper()
                    results.append(clean)
        
        return results
    
    def process_fasta(input_path, output_path, name):
        print(f"\n[Processing {name}]")
        records = list(SeqIO.parse(input_path, "fasta"))
        print(f"  Loaded {len(records)} sequences")
        
        # Extract sequences (truncate very long ones)
        MAX_LEN = 1000
        seqs = [str(r.seq)[:MAX_LEN] for r in records]
        ids = [r.id for r in records]
        
        # Generate 3Di
        print(f"  Generating 3Di sequences...")
        di3_seqs = seq_to_3di(seqs)
        
        # Write FASTA
        print(f"  Writing to {output_path}...")
        with open(output_path, "w") as f:
            for pid, di3 in zip(ids, di3_seqs):
                f.write(f">{pid}\n{di3}\n")
        
        print(f"  Done! Wrote {len(di3_seqs)} sequences.")
        return output_path
    
    # Process Train
    print("\n[2/4] Processing Training Set...")
    train_out = process_fasta(
        "/root/data/Train/train_sequences.fasta",
        "/root/train_3di.fasta",
        "Train"
    )
    
    # Process Test
    print("\n[3/4] Processing Test Set...")
    test_out = process_fasta(
        "/root/data/Test/testsuperset.fasta",
        "/root/test_3di.fasta",
        "Test"
    )
    
    # Read and return files
    print("\n[4/4] Reading output files...")
    with open(train_out, "r") as f:
        train_data = f.read()
    with open(test_out, "r") as f:
        test_data = f.read()
    
    return train_data, test_data

@app.local_entrypoint()
def main():
    print("🚀 Starting ProstT5 3Di Generation on Modal...")
    train_3di, test_3di = generate_3di_sequences.remote()
    
    # Save locally
    with open("train_3di.fasta", "w") as f:
        f.write(train_3di)
    print("✅ Saved train_3di.fasta")
    
    with open("test_3di.fasta", "w") as f:
        f.write(test_3di)
    print("✅ Saved test_3di.fasta")
    
    print("\n🎉 Done! Next steps:")
    print("  1. diamond makedb --in train_3di.fasta -d train_3di_db")
    print("  2. diamond blastp -q test_3di.fasta -d train_3di_db --custom-matrix 3di_matrix.mat ...")
