import modal
import os

app = modal.App("cafa-6-text-mining")

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
        "scikit-learn"
    )
    .add_local_file("Train/train_sequences.fasta", "/root/data/Train/train_sequences.fasta")
    .add_local_file("Test/testsuperset.fasta", "/root/data/Test/testsuperset.fasta")
    .add_local_file("Train/train_terms.tsv", "/root/data/Train/train_terms.tsv")
    .add_local_file("Train/go-basic.obo", "/root/data/Train/go-basic.obo")
    .add_local_file("uniprot_descriptions.tsv", "/root/data/uniprot_descriptions.tsv")
)

@app.function(image=image, gpu="A10G", timeout=86400)
def train_text_model():
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
    print("🏆 CAFA-6: Text Mining Model (Running on Modal A10G)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # CONFIG
    CONFIG = {
        'hidden_dims': [1024, 512, 256],
        'dropout': 0.3,
        'lr': 3e-4,
        'epochs': 15,
        'batch_size': 32, # Batch size for text model
        'n_folds': 5,
        'n_terms': 1000,
        'text_model': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        'max_text_len': 256,
        'data_dir': "/root/data"
    }
    
    # LOAD DATA
    print("\n[1/5] Loading Data...")
    
    # Load GO hierarchy
    go_graph = obonet.read_obo(f"{CONFIG['data_dir']}/Train/go-basic.obo")
    
    # Load Descriptions
    print("Loading descriptions...")
    desc_df = pd.read_csv(f"{CONFIG['data_dir']}/uniprot_descriptions.tsv", sep='\t')
    uniprot_desc = dict(zip(desc_df['accession'], desc_df['description']))
    
    # Load Proteins (to get list)
    train_prots = []
    for r in SeqIO.parse(f"{CONFIG['data_dir']}/Train/train_sequences.fasta", "fasta"):
        acc = r.id.split('|')[1] if '|' in r.id else r.id
        train_prots.append(acc)
        
    test_prots = []
    for r in SeqIO.parse(f"{CONFIG['data_dir']}/Test/testsuperset.fasta", "fasta"):
        acc = r.id.split('|')[1] if '|' in r.id else r.id
        test_prots.append(acc)
        
    print(f"Train: {len(train_prots):,}, Test: {len(test_prots):,}")
    
    # Load Labels
    df_terms = pd.read_csv(f"{CONFIG['data_dir']}/Train/train_terms.tsv", sep='\t', names=['EntryID', 'term', 'aspect'])
    if df_terms.iloc[0,0] == 'EntryID': df_terms = df_terms.iloc[1:]
    protein_to_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()
    
    # GENERATE TEXT EMBEDDINGS
    print("\n[2/5] Generating PubMedBERT Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['text_model'])
    model = AutoModel.from_pretrained(CONFIG['text_model']).to(device).eval()
    
    def get_text_embeddings(accs, batch_size=64):
        embeddings = []
        texts = []
        for acc in accs:
            # Augment with GO terms for training if available (simplified for Modal: just desc)
            texts.append(uniprot_desc.get(acc, f"Protein {acc}")[:1000])
            
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG['max_text_len'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :] # CLS token
                embeddings.append(emb.cpu().numpy())
            
            if i % 1000 == 0: print(f"  {i}/{len(texts)}")
            
        return np.vstack(embeddings)

    print("  Encoding Train...")
    X_train = get_text_embeddings(train_prots)
    print("  Encoding Test...")
    X_test = get_text_embeddings(test_prots)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # MODEL TRAINING
    print("\n[3/5] Training Classifiers...")
    
    class TextClassifier(nn.Module):
        def __init__(self, input_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, n_classes),
                nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)

    all_preds = []
    
    for aspect in ['F', 'P', 'C']:
        print(f"  Aspect: {aspect}")
        aspect_df = df_terms[df_terms['aspect'] == aspect]
        terms = aspect_df['term'].value_counts().head(CONFIG['n_terms']).index.tolist()
        term_to_idx = {t: i for i, t in enumerate(terms)}
        
        y_train = np.zeros((len(train_prots), len(terms)), dtype=np.float32)
        for i, acc in enumerate(train_prots):
            if acc in protein_to_terms:
                for term in protein_to_terms[acc]:
                    if term in term_to_idx:
                        y_train[i, term_to_idx[term]] = 1
                        
        # Train fold (Simplified: single fold for speed on Modal demo)
        model = TextClassifier(768, len(terms)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        model.train()
        for epoch in range(5): # Short training
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
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(test_tensor).cpu().numpy()
            
        for i, acc in enumerate(test_prots):
            for j, score in enumerate(preds[i]):
                if score > 0.01:
                    all_preds.append((acc, terms[j], score))
                    
    # SAVE
    print("\n[4/5] Saving Predictions...")
    df_out = pd.DataFrame(all_preds, columns=['protein', 'term', 'score'])
    return df_out.to_csv(index=False, sep='\t', header=False)

@app.local_entrypoint()
def main():
    print("🚀 Starting Modal Job...")
    csv_content = train_text_model.remote()
    
    with open("submission_text_mining.tsv", "w") as f:
        f.write(csv_content)
        
    print("✅ Done! Saved to submission_text_mining.tsv")
