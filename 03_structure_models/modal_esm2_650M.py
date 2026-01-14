import modal
import os

app = modal.App("cafa-6-structure-model")

# Define the environment & COPY FILES
# We need transformers for ESM2
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "transformers", 
        "biopython", 
        "obonet", 
        "pandas", 
        "numpy", 
        "scikit-learn"
    )
    .add_local_file("Train/train_sequences.fasta", "/root/data/Train/train_sequences.fasta")
    .add_local_file("Test/testsuperset.fasta", "/root/data/Test/testsuperset.fasta")
    .add_local_file("Train/train_terms.tsv", "/root/data/Train/train_terms.tsv")
    .add_local_file("Train/go-basic.obo", "/root/data/Train/go-basic.obo")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,  # 24 hours (was 12)400)
)
def train_structure_model():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
    from Bio import SeqIO
    import obonet
    from sklearn.model_selection import StratifiedKFold
    import gc
    
    print("="*80)
    print("🏆 CAFA-6: Structure Model (ESM2-650M) on Modal A10G")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # CONFIG
    CONFIG = {
        'hidden_dims': [1024, 512, 256],
        'dropout': 0.3,
        'lr': 5e-4,
        'epochs': 15,
        'batch_size': 32, # Batch size for Embeddings
        'n_folds': 5,
        'n_terms': 1000,
        'esm_model': "facebook/esm2_t33_650M_UR50D",
        'max_seq_len': 1022,
        'data_dir': "/root/data"
    }
    
    # LOAD DATA
    print("\n[1/5] Loading Data...")
    
    # Load GO hierarchy
    go_graph = obonet.read_obo(f"{CONFIG['data_dir']}/Train/go-basic.obo")
    
    # Load Proteins
    train_proteins = {}
    for r in SeqIO.parse(f"{CONFIG['data_dir']}/Train/train_sequences.fasta", "fasta"):
        acc = r.id.split('|')[1] if '|' in r.id else r.id
        train_proteins[acc] = str(r.seq)
        
    test_proteins = {}
    for r in SeqIO.parse(f"{CONFIG['data_dir']}/Test/testsuperset.fasta", "fasta"):
        acc = r.id.split('|')[1] if '|' in r.id else r.id
        test_proteins[acc] = str(r.seq)
        
    print(f"Train: {len(train_proteins):,}, Test: {len(test_proteins):,}")
    
    # Load Labels
    df_terms = pd.read_csv(f"{CONFIG['data_dir']}/Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
    if df_terms.iloc[0,0] == 'EntryID': df_terms = df_terms.iloc[1:]
    protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
    
    # GENERATE EMBEDDINGS
    print("\n[2/5] Generating ESM2 Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['esm_model'])
    model = AutoModel.from_pretrained(CONFIG['esm_model']).to(device).eval()
    
    def get_embeddings(proteins, batch_size=32):
        embeddings = []
        accs = list(proteins.keys())
        seqs = [proteins[a][:CONFIG['max_seq_len']] for a in accs]
            
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG['max_seq_len']+2)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1) # Average pooling
                embeddings.append(emb.cpu().numpy())
            
            if i % 1000 < batch_size: print(f"  {i}/{len(seqs)}")
            
        return accs, np.vstack(embeddings)

    print("  Encoding Train...")
    train_accs, X_train = get_embeddings(train_proteins, batch_size=24) # Slightly smaller batch for 650M
    print("  Encoding Test...")
    test_accs, X_test = get_embeddings(test_proteins, batch_size=24)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # MODEL TRAINING
    print("\n[3/5] Training Classifiers...")
    
    class WinnerStyleModel(nn.Module):
        def __init__(self, input_dim, hidden_dims, n_classes):
            super().__init__()
            layers = []
            dims = [input_dim] + hidden_dims
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(dims[-1], n_classes))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.net(x)

    all_preds = []
    
    for aspect in ['F', 'P', 'C']:
        print(f"  Aspect: {aspect}")
        aspect_df = df_terms[df_terms['aspect'] == aspect]
        terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
        term_to_idx = {t: i for i, t in enumerate(terms)}
        
        y_train = np.zeros((len(train_accs), len(terms)), dtype=np.float32)
        train_acc_set = set(train_accs)
        train_acc_map = {acc: i for i, acc in enumerate(train_accs)}
        
        for acc, p_terms in protein_to_terms.items():
            if acc in train_acc_map:
                for term in p_terms:
                    if term in term_to_idx:
                        y_train[train_acc_map[acc], term_to_idx[term]] = 1
                        
        # Train fold
        model = WinnerStyleModel(1280, CONFIG['hidden_dims'], len(terms)).to(device) # ESM2-650M is 1280d
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(10): # Efficient training
            total_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"    Epoch {epoch}: {total_loss/len(loader):.4f}")
            
        model.eval()
        with torch.no_grad():
            # Batch infer test
            test_preds = []
            test_tensor = torch.tensor(X_test, dtype=torch.float32)
            for i in range(0, len(test_tensor), 1000):
                 batch = test_tensor[i:i+1000].to(device)
                 test_preds.append(model(batch).cpu().numpy())
            preds = np.vstack(test_preds)
            
        for i, acc in enumerate(test_accs):
            for j, score in enumerate(preds[i]):
                if score > 0.01:
                    all_preds.append((acc, terms[j], score))
                    
    # SAVE
    print("\n[4/5] Saving Predictions...")
    df_out = pd.DataFrame(all_preds, columns=['protein', 'term', 'score'])
    return df_out.to_csv(index=False, sep='\t', header=False)

@app.local_entrypoint()
def main():
    print("🚀 Starting Modal Structure Model (ESM2)...")
    csv_content = train_structure_model.remote()
    
    with open("submission_structure.tsv", "w") as f:
        f.write(csv_content)
        
    print("✅ Done! Saved to submission_structure.tsv")
