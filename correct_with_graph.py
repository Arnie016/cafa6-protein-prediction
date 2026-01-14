import pandas as pd
import obonet
import networkx as nx
import argparse
import sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Graph Diffusion for GO (Deep Clean)")
    parser.add_argument('--input', required=True, help="Input prediction TSV")
    parser.add_argument('--output', required=True, help="Output prediction TSV")
    parser.add_argument('--obo', default='Train/go-basic.obo', help="Path to go-basic.obo")
    parser.add_argument('--alpha', type=float, default=0.5, help="Diffusion factor (0.0-1.0)")
    args = parser.parse_args()

    print(f"Loading GO Graph from {args.obo}...")
    go_graph = obonet.read_obo(args.obo)
    
    # Invert the graph: We want to propagate score from Child -> Parent.
    # obonet: Child --is_a--> Parent.
    # So networkx edges are (Child, Parent).
    # We want flow: Child -> Parent.
    
    # Create an optimized adjacency list
    parents_of = defaultdict(list)
    for node in go_graph.nodes():
        if node in go_graph:
            # networkx successors in obonet are parents (is_a target)
            parents_of[node] = list(go_graph.successors(node))

    print(f"Loading predictions from {args.input}...")
    # Load all predictions (optimizing for speed)
    df = pd.read_csv(args.input, sep='\t', names=['protein', 'term', 'score'])
    
    # Filter low scores to save RAM? No, diffusion needs low scores too.
    # Group by protein
    print("Grouping by protein...")
    grouped = df.groupby('protein')
    
    results = []
    
    print("Applying Graph Diffusion (Consistency Propagation)...")
    
    # Parameters
    alpha = args.alpha
    
    for protein, group in tqdm(grouped, total=len(grouped)):
        # Convert to dict for O(1) access
        scores = defaultdict(float)
        for t, s in zip(group['term'], group['score']):
            scores[t] = s
            
        # 1. Standard Propagation (Child -> Parent Rule)
        # DeepGOZero style Logic Constraint: Score(p) >= Score(c)
        # We can do this topologically, or iteratively.
        # Iterative is robust for cyclic errors (though GO is DAG).
        
        # We perform a "Heat Diffusion"
        # Since we want to update Parents based on Children.
        # We verify if we need to add parents that are missing.
        
        # Get all terms currently predicted
        active_terms = list(scores.keys())
        
        # For each active term, propagate up
        # We use a set to track visited to avoid infinite loops
        # But doing this for every protein is slow.
        
        # FAST APPROXIMATION:
        # 1. Sort terms by score (descending) - Strongest evidence first.
        # 2. Propagate up.
        
        sorted_terms = sorted(active_terms, key=lambda x: scores[x], reverse=True)
        
        updated_scores = scores.copy()
        
        # Add Parents
        # If Child > Parent, set Parent = Child.
        # This is the "Hard Constraint".
        # "Soft Constraint" would be: Parent = Parent + alpha * Child (Diffusion).
        
        # Let's use Hard Constraint (Winner Strategy)
        visited = set()
        
        queue = sorted_terms # Start with high confidence nodes
        
        # Breadth-first propagation UP the graph
        # For each node in queue, push score to parents.
        
        # Simple Pass:
        # Check every predicted term. Ensure parents >= term.
        # Repeat until stable? (2 passes is usually enough)
        
        for _ in range(2): 
            # We iterate over a SNAPSHOT of keys to allow adding new parents
            curr_keys = list(updated_scores.keys())
            
            for term in curr_keys:
                child_score = updated_scores[term]
                if child_score < 0.01: continue
                
                # Get parents
                parents = parents_of.get(term, [])
                
                for parent in parents:
                    parent_score = updated_scores[parent] # Default 0.0
                    
                    # LOGIC: Parent must be at least as probable as child
                    if parent_score < child_score:
                        updated_scores[parent] = child_score
                        # Note: We just added 'parent' to updated_scores if it wasn't there.
                        # This implicitly expands the prediction set!
        
        # Save results
        # Only keep significant scores
        for t, s in updated_scores.items():
            if s >= 0.001: # Filter tiny values
                results.append((protein, t, f"{s:.6f}"))

    print("Saving diffused predictions...")
    df_out = pd.DataFrame(results, columns=['protein', 'term', 'score'])
    df_out.to_csv(args.output, sep='\t', index=False, header=False)
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
