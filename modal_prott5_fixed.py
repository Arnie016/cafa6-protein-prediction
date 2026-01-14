import modal
import os

app = modal.App("cafa-6-structure-model-3B")

# Define the environment & COPY FILES
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "transformers", 
        "biopython", 
        "obonet", 
        "pandas", 
        "numpy", 
        "scikit-learn",
        "sentencepiece"
    )
    .add_local_file("Train/train_sequences.fasta", "/root/data/Train/train_sequences.fasta")
    .add_local_file("Test/testsuperset.fasta", "/root/data/Test/testsuperset.fasta")
    .add_local_file("Train/train_terms.tsv", "/root/data/Train/train_terms.tsv")
    .add_local_file("Train/go-basic.obo", "/root/data/Train/go-basic.obo")
)

# REQUESTING A100 GPU (40GB VRAM minimum for 3B parameter model)
@app.function(image=image, gpu="A100", timeout=86400)
def train_structure_model_3B():
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
    print("🏆 CAFA-6: HEAVY Structure Model (ESM2-3B) on Modal A100")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # CONFIG
    CONFIG = {
        'hidden_dims': [1024, 512, 256],
        'dropout': 0.3,
        'lr': 1e-4, 
        'epochs': 10,
        'batch_size': 8, 
        'n_folds': 3,
        'n_terms': 1000,
        'model_name': "Rostlab/prot_t5_xl_uniref50", # GOOGLE MODEL
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
    print("\n[2/5] Generating ProtT5-XL Embeddings...")
    from transformers import T5Tokenizer, T5EncoderModel
    
    tokenizer = T5Tokenizer.from_pretrained(CONFIG['model_name'], do_lower_case=False)
    model = T5EncoderModel.from_pretrained(CONFIG['model_name']).to(device).eval()
    
    def get_embeddings(proteins, batch_size=8): 
        embeddings = []
        accs = list(proteins.keys())
        # ProtT5 expects space-separated amino acids: "M A L ..."
        seqs = [" ".join(list(proteins[a][:CONFIG['max_seq_len']])) for a in accs]
            
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            # T5 Tokenizer usage
            inputs = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest", return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # T5 embeddings: last_hidden_state (Batch, Len, 1024)
                # Mean pool excluding padding
                emb = outputs.last_hidden_state.mean(dim=1) 
                embeddings.append(emb.cpu().numpy())
            
            if i % 1000 < batch_size: print(f"  {i}/{len(seqs)}")
            
        return accs, np.vstack(embeddings)

    print("  Encoding Train (ProtT5)...")
    train_accs, X_train = get_embeddings(train_proteins, batch_size=6) # XL is heavier than 3B sometimes
    print("  Encoding Test (ProtT5)...")
    test_accs, X_test = get_embeddings(test_proteins, batch_size=6)
    
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
        model = WinnerStyleModel(1024, CONFIG['hidden_dims'], len(terms)).to(device) # ProtT5 is 1024d
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(CONFIG['epochs']):
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
    print("🚀 Starting Modal Structure Model (ProtT5-XL)...")
    csv_content = train_structure_model_3B.remote()
    
    with open("submission_prott5.tsv", "w") as f:
        f.write(csv_content)
        
    print("✅ Done! Saved to submission_structure_3B.tsv")
