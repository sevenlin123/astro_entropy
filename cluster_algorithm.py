import os
import re
import glob
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def standardize_algorithm_text(text):
    """
    Removes arbitrary list formatting and standardizes text for embedding.
    """
    standardized = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    standardized = standardized.replace('\n', ' ')
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized.lower()

def cluster_algorithms(cleaned_algorithms, source_labels, model_name):
    """
    Visualizes the semantic space separating all models across information states.
    """
    # 1. Load the embedding model (code specialized)
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code", 
        trust_remote_code=True,
        device="mps"
    )
    
    # 2. Generate embeddings
    print("Generating embeddings...")
    model.max_seq_length = 512
    embeddings = model.encode(cleaned_algorithms, batch_size=8, show_progress_bar=True)

    # 3. Calculate raw distance matrix
    distances = cosine_distances(embeddings)
    
    # 4. UMAP Dimensionality Reduction
    print("Applying UMAP...")
    STATIC_NEIGHBORS = 5
    if len(cleaned_algorithms) <= STATIC_NEIGHBORS:
        raise ValueError(f"Sample size too small for n_neighbors={STATIC_NEIGHBORS}.")
    
    reducer = umap.UMAP(n_neighbors=STATIC_NEIGHBORS, n_components=2, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # 5. Visual Cluster mapping (ignoring HDBSCAN for this visualization view)

    # --- VISUALIZATION BLOCK ---
    if len(cleaned_algorithms) > 1:
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        
        # ---------------------------------------------------------
        # DEFINING STYLES: Specific Colors & Shapes for EVERY combination
        # Colors grouped by state: T (Blues/Purples), TA (Oranges/Reds), TAM (Greens)
        # Shapes grouped by model: Local(o), Gemini(*), GPT(D), Claude(X)
        # ---------------------------------------------------------
        styles = {
            # --- PRIOR STATE (T) ---
            f'{model_name}_T': {'c': '#1f77b4', 'm': 'o', 's': 50, 'lab': f'{model_name} (T)'}, # Standard Blue
            'Gemini_T':         {'c': '#9edae5', 'm': '*', 's': 350, 'lab': 'Gemini (T)'},        # Light Blue star
            'GPT_T':            {'c': '#9467bd', 'm': 'D', 's': 180, 'lab': 'GPT-5 (T)'},         # Purple diamond
            'Claude_T':         {'c': '#c5b0d5', 'm': 'X', 's': 200, 'lab': 'Claude (T)'},        # Lavender X
            
            # --- INTERMEDIATE STATE (TA) ---
            f'{model_name}_TA': {'c': '#ff7f0e', 'm': 'o', 's': 50, 'lab': f'{model_name} (TA)'}, # Standard Orange
            'Gemini_TA':        {'c': '#ffbb78', 'm': '*', 's': 350, 'lab': 'Gemini (TA)'},       # Light Orange star
            'GPT_TA':           {'c': '#d62728', 'm': 'D', 's': 180, 'lab': 'GPT-5 (TA)'},        # Red diamond
            'Claude_TA':        {'c': '#ff9896', 'm': 'X', 's': 200, 'lab': 'Claude (TA)'},       # Pink X

            # --- POSTERIOR STATE (TAM) ---
            f'{model_name}_TAM':{'c': '#2ca02c', 'm': 'o', 's': 50, 'lab': f'{model_name} (TAM)'},# Standard Green
            'Gemini_TAM':       {'c': '#b5ffb9', 'm': '*', 's': 350, 'lab': 'Gemini (TAM)'},      # Lime Green star
            'GPT_TAM':          {'c': '#8c564b', 'm': 'D', 's': 180, 'lab': 'GPT-5 (TAM)'},       # Brown diamond
            'Claude_TAM':       {'c': '#c49c94', 'm': 'X', 's': 200, 'lab': 'Claude (TAM)'}        # Tan X
        }
        
        # Plot 1: UMAP Scatter Plot (Granular separation)
        unique_source_labels = sorted(list(set(source_labels))) # Sorted T -> TA -> TAM
        
        for slabel in unique_source_labels:
            if slabel not in styles: continue
            
            # Find indices of points belonging to this specific model+state
            idx = [i for i, src in enumerate(source_labels) if src == slabel]
            if not idx: continue
            
            style = styles[slabel]
            ax1.scatter(
                umap_embeddings[idx, 0], 
                umap_embeddings[idx, 1], 
                c=style['c'], 
                marker=style['m'],
                s=style['s'], 
                label=f"{style['lab']}",
                edgecolor='k', 
                linewidth=0.5,
                zorder=10 if style['m'] != 'o' else 1, # API models to front
                alpha = 0.9 if style['m'] != 'o' else 0.5
            )
            
        ax1.set_title(f'UMAP Semantic Space: Local vs. Separated API Models', fontsize=16, fontweight='bold')
        ax1.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax1.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Create a cleaner custom legend grouping by state
        ax1.legend(fontsize=11, loc='best', ncol=2, title="Model & Information State")
    '''
        # Plot 2: Pairwise Cosine Distance Heatmap (Granular Separation)
        im = ax2.imshow(distances, cmap='Blues', vmin=0, vmax=0.3)
        cbar = plt.colorbar(im, ax=ax2, pad=0.02)
        cbar.set_label(label='Cosine Distance', size=14)
        
        # Draw dashed lines for main state boundaries (T, TA, TAM)
        len_T = sum(1 for s in source_labels if '_T' in s)
        len_TA = sum(1 for s in source_labels if '_TA' in s)
        
        if len_T > 0:
            ax2.axhline(len_T - 0.5, color='black', linewidth=2, linestyle='--')
            ax2.axvline(len_T - 0.5, color='black', linewidth=2, linestyle='--')
        if len_TA > 0:
            ax2.axhline(len_T + len_TA - 0.5, color='black', linewidth=2, linestyle='--')
            ax2.axvline(len_T + len_TA - 0.5, color='black', linewidth=2, linestyle='--')
        
        # Format axes to show specific source labels
        ticks = np.arange(len(cleaned_algorithms))
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        
        # Shorten labels for the heatmap axes to fit
        short_labels = [styles[s]['lab'].replace(' (','\n(') if s in styles else s for s in source_labels]
        ax2.set_xticklabels(short_labels, rotation=90, fontsize=8, ha='center')
        ax2.set_yticklabels(short_labels, fontsize=8, va='center')
        ax2.set_title('Pairwise Cosine Distances Across Models & States', fontsize=16, fontweight='bold')
    '''    
    plt.tight_layout()
    plt.savefig('separated_model_cluster.eps')
    plt.show()
    
    return source_labels

def main():
    model = 'deepseek' # Choose 'deepseek' or 'gpt-oss'

    # 1. Setup paths for Local Model
    local_paths = {
        'T': sorted(glob.glob(f'{model}/algorithm_pseudocode_t/*txt')),
        'TA': sorted(glob.glob(f'{model}/algorithm_pseudocode_ta/*txt')),
        'TAM': sorted(glob.glob(f'{model}/algorithm_pseudocode_tam/*txt'))
    }
    
    # 2. Setup paths for API Models (Fixed Typos from Turn 12)
    api_map = {
        'Gemini': {'T': 'gemini_t.algo.txt', 'TA': 'gemini_ta.algo.txt', 'TAM': 'gemini_tam.algo.txt'},
        'GPT':    {'T': 'gpt_T_algo.txt',    'TA': 'gpt_TA_algo.txt',    'TAM': 'gpt_TAM_algo.txt'},
        'Claude': {'T': 'claude_T_algo.txt', 'TA': 'claude_TA_algo.txt', 'TAM': 'claude_TAM_algo.txt'} # Fixed paths assuming they exist
    }

    algor_names = []
    source_labels = []

    # ---------------------------------------------------------
    # 3. Combine Data: Grouped by state (T -> TA -> TAM) 
    # and then by specific model within that state.
    # ---------------------------------------------------------
    for state in ['T', 'TA', 'TAM']:
        # Append Local Model data first for this state
        for fname in local_paths[state]:
            algor_names.append(fname)
            source_labels.append(f'{model}_{state}')
        
        # Append API Models data for this state
        for api_name, states_map in api_map.items():
            fpath = os.path.join('api_results', states_map[state])
            if os.path.exists(fpath):
                algor_names.append(fpath)
                source_labels.append(f'{api_name}_{state}')
    
    # 4. Load standard text
    algorithms = [standardize_algorithm_text(open(i, 'r').read()) for i in algor_names]
    
    if not algorithms:
        print("No text files found. Check your paths.")
        return
        
    print(f"Loaded {len(algorithms)} algorithms from Local ({model}) and API models.")
    
    # Run visualization
    _ = cluster_algorithms(algorithms, source_labels, model)

if __name__ == "__main__":
    main()