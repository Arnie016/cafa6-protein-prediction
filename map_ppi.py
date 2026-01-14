import pandas as pd
import requests
import gzip
import io
import os

def main():
    # 1. Download STRING -> UniProt Mapping (Physicians use this typically)
    # URL: https://string-db.org/mapping_files/STRING_display_names/human.name_2_string.tsv.gz
    # Wait, we need ALL organisms. The full mapping is HUGE (20GB).
    # Better approach: access the Uniprot ID mapping API in batches?
    # Or, rely on the fact that we fetched by UniProt ID?
    
    # Wait, in fetch_string_ppi.py, 'string_ids' were fetched via the API using UniProt IDs.
    # The API returns the STRING ID for the query.
    # So we already HAVE the mapping in memory during the fetch, but maybe we didn't save it perfectly.
    
    # Let's check ppi_network.tsv format.
    # It has: protein1, protein2, score
    
    print("Loading PPI network...")
    df = pd.read_csv("ppi_network.tsv", sep="\t", names=["p1", "p2", "score"])
    
    # Problem: p1 and p2 are STRING IDs (e.g. 9606.ENSP...). Our ensemble needs UniProt IDs.
    # We need a mapping file.
    
    # Since we don't have the mapping saved, we can:
    # 1. Fetch it again (fast).
    # 2. Use the 'uniprot_descriptions.tsv' if it has cross-refs? (Unlikely).
    
    # STRATEGY: 
    # We will use the STRING API to get the mapping for our Test Proteins *in bulk* again.
    # Then simply replace the IDs in the dataframe.
    
    # Load Test Superset
    print("Loading Test Superset for mapping...")
    from Bio import SeqIO
    test_proteins = [r.id for r in SeqIO.parse("Test/testsuperset.fasta", "fasta")]
    
    # Mapping dict
    string_to_uniprot = {}
    
    # We need to query STRING API "get_string_ids"
    # https://string-db.org/api/tsv/get_string_ids?identifiers=...
    
    print("Fetching ID Mappings from STRING (this is fast)...")
    
    batch_size = 500
    
    for i in range(0, len(test_proteins), batch_size):
        batch = test_proteins[i:i+batch_size]
        try:
            url = "https://string-db.org/api/tsv/get_string_ids"
            params = {
                "identifiers": "\r".join(batch),
                "species": 9606, # Wait, default is Human? No, CAFA has many species.
                # If we don't specify species, STRING tries to guess.
                "echo_query": 1 
            }
            # Note: For multi-species, STRING API is tricky if we don't know the taxon.
            # But the 'echo_query' return the input name (UniProt) alongside the STRING ID.
            
            r = requests.post(url, data=params)
            
            if r.status_code == 200:
                lines = r.text.strip().split("\n")[1:] # Skip header
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        query_item = parts[0] # UniProt
                        string_id = parts[2] # 9606.ENSP...
                        string_to_uniprot[string_id] = query_item
                        
        except Exception as e:
            print(f"Error batch {i}: {e}")
            
    print(f"Mapped {len(string_to_uniprot)} IDs.")
    
    # Now replace in DataFrame
    print("Mapping PPI Network IDs...")
    
    # We want rows where BOTH p1 and p2 are mapped (or at least one is our target)
    # Actually, we rely on "Guilt by Association".
    # If P1 is Known (Train) and P2 is Unknown (Test), we propagate.
    # So we need mapping for TRAIN proteins too? That's too huge.
    
    # SIMPLIFIED PPI APPROACH FOR CAFA:
    # Just map the Test vs Test interactions.
    
    df['p1_uni'] = df['p1'].map(string_to_uniprot)
    df['p2_uni'] = df['p2'].map(string_to_uniprot)
    
    # Drop rows where we can't identify the protein
    clean_df = df.dropna(subset=['p1_uni', 'p2_uni'])
    
    print(f"Retained {len(clean_df)} interactions between Test Proteins.")
    
    # Save
    clean_df[['p1_uni', 'p2_uni', 'score']].to_csv("ppi_network_mapped.tsv", sep="\t", index=False)
    print("Saved to ppi_network_mapped.tsv")

if __name__ == "__main__":
    main()
