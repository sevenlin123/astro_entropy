import os
import re
import glob
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools 

# NEW IMPORTS FOR METRICS & CLUSTERING
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, cosine

def standardize_algorithm_text(text):
    """
    Removes arbitrary list formatting and standardizes text for embedding.
    """
    standardized = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    standardized = standardized.replace('\n', ' ')
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized.lower()

# ==========================================
# MATHEMATICAL METRICS FUNCTIONS
# ==========================================
def calc_entropy(prob_dist):
    """Calculates Shannon Entropy in bits."""
    return entropy(prob_dist, base=2)

def calc_jsd(p, q):
    """Calculates Jensen-Shannon Divergence in bits."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.sum(p) == 0 or np.sum(q) == 0:
        return np.nan 
    p = p / np.sum(p)
    q = q / np.sum(q)
    return jensenshannon(p, q, base=2) ** 2

# ==========================================
# BOOTSTRAPPING FUNCTIONS (SEMANTIC)
# ==========================================
def bootstrap_semantic_entropy(indices, cluster_labels, num_clusters=15, n_iterations=1000):
    """Calculates 95% CI for Semantic Entropy via bootstrapping."""
    if not indices or len(indices) == 1:
        counts = np.bincount(cluster_labels[indices], minlength=num_clusters) if indices else np.zeros(num_clusters)
        prob = counts / np.sum(counts) if np.sum(counts) > 0 else np.zeros(num_clusters)
        val = calc_entropy(prob)
        return val, val, val

    n_docs = len(indices)
    results = []
    for _ in range(n_iterations):
        sampled_idx = np.random.choice(indices, size=n_docs, replace=True)
        counts = np.bincount(cluster_labels[sampled_idx], minlength=num_clusters)
        prob = counts / np.sum(counts)
        results.append(calc_entropy(prob))

    return np.mean(results), np.percentile(results, 2.5), np.percentile(results, 97.5)

def bootstrap_semantic_jsd(indices_p, indices_q, cluster_labels, num_clusters=15, n_iterations=1000):
    """Calculates 95% CI for Semantic JSD via bootstrapping."""
    if not indices_p or not indices_q:
        return np.nan, np.nan, np.nan

    n_p = len(indices_p)
    n_q = len(indices_q)

    # Pre-compute q if it's a single item (e.g., GT) to avoid resampling variance
    prob_q_fixed = None
    if n_q == 1:
        counts_q = np.bincount(cluster_labels[indices_q], minlength=num_clusters)
        prob_q_fixed = counts_q / np.sum(counts_q)

    results = []
    for _ in range(n_iterations):
        samp_p = np.random.choice(indices_p, size=n_p, replace=True)
        counts_p = np.bincount(cluster_labels[samp_p], minlength=num_clusters)
        prob_p = counts_p / np.sum(counts_p)

        if prob_q_fixed is not None:
            prob_q = prob_q_fixed
        else:
            samp_q = np.random.choice(indices_q, size=n_q, replace=True)
            counts_q = np.bincount(cluster_labels[samp_q], minlength=num_clusters)
            prob_q = counts_q / np.sum(counts_q)

        results.append(calc_jsd(prob_p, prob_q))

    return np.mean(results), np.percentile(results, 2.5), np.percentile(results, 97.5)

def visualize_unified_semantic_space(cleaned_algorithms, source_labels):
    """
    Embeds algorithms, applies UMAP, calculates metrics, and plots.
    """
    # 1. Load the embedding model
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code", 
        trust_remote_code=True,
        device="mps" # Change to 'cuda' or 'cpu' if not on Apple Silicon
    )
    
    # 2. Generate embeddings
    print(f"Generating embeddings for {len(cleaned_algorithms)} total algorithms...")
    model.max_seq_length = 512
    embeddings = model.encode(cleaned_algorithms, batch_size=8, show_progress_bar=True)
    
    # =========================================================
    # QUANTITATIVE METRICS OUTPUT (Semantic Space)
    # =========================================================
    print("\n" + "="*60)
    print(" EXPERIMENT 1: QUANTITATIVE METRICS (SEMANTIC SPACE)")
    print("="*60)

    # Set seed for exactly reproducible confidence intervals
    np.random.seed(42)

    # A. Cluster the continuous space to create discrete probability bins (for Entropy/JSD)
    NUM_CLUSTERS = 15 # Defines the number of discrete semantic classes
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # B. Isolate the Ground Truth continuous vector (for Cosine Distance)
    gt_idx = source_labels.index('Ground_Truth') if 'Ground_Truth' in source_labels else -1
    gt_emb = embeddings[gt_idx] if gt_idx != -1 else None

    def get_indices(target_label):
        """Helper to fetch document indices for a specific model/state."""
        return [i for i, src in enumerate(source_labels) if src == target_label]

    def get_mean_emb(target_label):
        """Calculates the geometric centroid (mean vector) for a given state."""
        indices = get_indices(target_label)
        if not indices: return None
        return np.mean(embeddings[indices], axis=0)

    # --- LOCAL MODELS METRICS ---
    for l_model in ['gpt-oss', 'deepseek']:
        if f"{l_model}_T" not in source_labels: continue
        
        print(f"\nModel: {l_model.upper()}")
        print("-" * 60)
        
        idx_T = get_indices(f'{l_model}_T')
        idx_TA = get_indices(f'{l_model}_TA')
        idx_TAM = get_indices(f'{l_model}_TAM')
        
        emb_T = get_mean_emb(f'{l_model}_T')
        emb_TA = get_mean_emb(f'{l_model}_TA')
        emb_TAM = get_mean_emb(f'{l_model}_TAM')
        
        # 1. Shannon Entropies (Semantic Uncertainty) [Mean, 95% CI]
        print("1. Semantic Entropy H_emb(Y|X) (bits) [Mean, 95% CI]:")
        for state, idx_list in [('T', idx_T), ('TA', idx_TA), ('TAM', idx_TAM)]:
            mean_val, low, high = bootstrap_semantic_entropy(idx_list, cluster_labels, NUM_CLUSTERS)
            print(f"   {state:<3} : {mean_val:.4f} [{low:.4f}, {high:.4f}]")
        print()
        
        # 2. Inter-State Divergence (Semantic Shift - JSD) [Mean, 95% CI]
        print("2. Inter-State Divergence (JSD in bits) [Mean, 95% CI]:")
        transitions = [('T -> TA', idx_T, idx_TA), 
                       ('TA -> TAM', idx_TA, idx_TAM), 
                       ('T -> TAM', idx_T, idx_TAM)]
        for label, p_idx, q_idx in transitions:
            mean_val, low, high = bootstrap_semantic_jsd(p_idx, q_idx, cluster_labels, NUM_CLUSTERS)
            print(f"   {label:<9}: {mean_val:.4f} [{low:.4f}, {high:.4f}]")
        print()
        
        # 3. Ground-Truth Alignment (Cosine Distance)
        if gt_emb is not None:
            print("3. Ground-Truth Alignment (Cosine Distance to GT):")
            if emb_T is not None: print(f"   T   vs GT : {cosine(emb_T, gt_emb):.4f}")
            if emb_TA is not None: print(f"   TA  vs GT : {cosine(emb_TA, gt_emb):.4f}")
            if emb_TAM is not None: print(f"   TAM vs GT : {cosine(emb_TAM, gt_emb):.4f}")
            print()
            
        # 4. Intra-Model Semantic Shift (Cosine Distance)
        print("4. Intra-Model Semantic Shift (Cosine Distance):")
        if emb_T is not None and emb_TA is not None:
            print(f"   T  -> TA  : {cosine(emb_T, emb_TA):.4f}")
        if emb_TA is not None and emb_TAM is not None:
            print(f"   TA -> TAM : {cosine(emb_TA, emb_TAM):.4f}")
        if emb_T is not None and emb_TAM is not None:
            print(f"   T  -> TAM : {cosine(emb_T, emb_TAM):.4f}")
            
        print("-" * 60)
        
    # --- API MODELS METRICS (GT, PAIRWISE, & INTRA-MODEL) ---
    print("\n" + "="*60)
    print(" API MODELS: ALIGNMENT & SHIFTS")
    print("="*60)
    
    api_models = ['Gemini', 'GPT-5', 'Claude']
    
    # 1. Distance to Ground Truth & Pairwise Distances per State
    for state in ['T', 'TA', 'TAM']:
        print(f"\n--- STATE: {state} ---")
        
        if gt_emb is not None:
            print("Cosine Distance to Ground Truth:")
            for api in api_models:
                api_emb = get_mean_emb(f"{api}_{state}")
                if api_emb is not None:
                    dist = cosine(api_emb, gt_emb)
                    print(f"   {api:<10} vs GT : {dist:.4f}")
        
        print("\nPairwise Cosine Distances between API Models:")
        for api1, api2 in itertools.combinations(api_models, 2):
            emb1 = get_mean_emb(f"{api1}_{state}")
            emb2 = get_mean_emb(f"{api2}_{state}")
            if emb1 is not None and emb2 is not None:
                dist = cosine(emb1, emb2)
                print(f"   {api1:<10} vs {api2:<10} : {dist:.4f}")

    # 2. Intra-Model Shifts Across States
    print("\n--- INTRA-MODEL SHIFTS ACROSS STATES (COSINE DISTANCE) ---")
    for api in api_models:
        print(f"\nModel: {api}")
        emb_T = get_mean_emb(f"{api}_T")
        emb_TA = get_mean_emb(f"{api}_TA")
        emb_TAM = get_mean_emb(f"{api}_TAM")
        
        if emb_T is not None and emb_TA is not None:
            print(f"   T  -> TA  : {cosine(emb_T, emb_TA):.4f}")
        if emb_TA is not None and emb_TAM is not None:
            print(f"   TA -> TAM : {cosine(emb_TA, emb_TAM):.4f}")
        if emb_T is not None and emb_TAM is not None:
            print(f"   T  -> TAM : {cosine(emb_T, emb_TAM):.4f}")

    print("="*60 + "\n")

    # 3. UMAP Dimensionality Reduction
    print("Applying UMAP for Visualization...")
    STATIC_NEIGHBORS = 5
    if len(cleaned_algorithms) <= STATIC_NEIGHBORS:
        raise ValueError(f"Sample size too small for n_neighbors={STATIC_NEIGHBORS}.")
    
    reducer = umap.UMAP(n_neighbors=STATIC_NEIGHBORS, n_components=2, metric='cosine', random_state=2)
    umap_embeddings = reducer.fit_transform(embeddings)

    # --- VISUALIZATION BLOCK ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_color('#e0e0e0')
    ax.spines['right'].set_color('#e0e0e0')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    
    styles = {
        'Ground_Truth': {'c': '#FFD700', 'm': 'P', 's': 500, 'z': 7},
        'gpt-oss_T':  {'c': '#1f77b4', 'm': 'o', 's': 60, 'z': 5},
        'deepseek_T': {'c': '#1f77b4', 'm': 's', 's': 60, 'z': 5},
        'Gemini_T':   {'c': '#1f77b4', 'm': '*', 's': 300, 'z': 8},
        'GPT-5_T':    {'c': '#1f77b4', 'm': 'D', 's': 120, 'z': 8},
        'Claude_T':   {'c': '#1f77b4', 'm': 'X', 's': 150, 'z': 8},
        'gpt-oss_TA':  {'c': '#ff7f0e', 'm': 'o', 's': 60, 'z': 5},
        'deepseek_TA': {'c': '#ff7f0e', 'm': 's', 's': 60, 'z': 5},
        'Gemini_TA':   {'c': '#ff7f0e', 'm': '*', 's': 600, 'z': 9},
        'GPT-5_TA':    {'c': '#ff7f0e', 'm': 'D', 's': 120, 'z': 9},
        'Claude_TA':   {'c': '#ff7f0e', 'm': 'X', 's': 150, 'z': 9},
        'gpt-oss_TAM':  {'c': '#2ca02c', 'm': 'o', 's': 60, 'z': 5},
        'deepseek_TAM': {'c': '#2ca02c', 'm': 's', 's': 60, 'z': 5},
        'Gemini_TAM':   {'c': '#2ca02c', 'm': '*', 's': 300, 'z': 10},
        'GPT-5_TAM':    {'c': '#2ca02c', 'm': 'D', 's': 120, 'z': 10},
        'Claude_TAM':   {'c': '#2ca02c', 'm': 'X', 's': 150, 'z': 10}
    }
    
    centroids = {'T': [], 'TA': [], 'TAM': []}
    
    for slabel in set(source_labels):
        idx = [i for i, src in enumerate(source_labels) if src == slabel]
        if not idx: continue
        
        style = styles.get(slabel, {'c': 'gray', 'm': 'o', 's': 50, 'z': 1})
        is_api = style['m'] not in ['o', 's']
        alpha_val = 0.9 if is_api else 0.3
        
        points_x = umap_embeddings[idx, 0]
        points_y = umap_embeddings[idx, 1]
        
        ax.scatter(
            points_x, points_y, 
            c=style['c'], marker=style['m'], s=style['s'], 
            zorder=style['z'], edgecolor='k' if is_api else 'none',
            linewidth=1.0 if is_api else 0, alpha=alpha_val
        )
        
        if '_T' in slabel and 'TA' not in slabel: centroids['T'].extend(zip(points_x, points_y))
        elif '_TA' in slabel and 'TAM' not in slabel: centroids['TA'].extend(zip(points_x, points_y))
        elif '_TAM' in slabel: centroids['TAM'].extend(zip(points_x, points_y))

    x_max = ax.get_xlim()[1]
    
    state_meta = {
        'T': {'name': 'Condition T\n(Title Only)', 'color': '#1f77b4'},
        'TA': {'name': 'Condition TA\n(Title + Abstract)', 'color': '#ff7f0e'},
        'TAM': {'name': 'Condition TAM\n(Full Text)', 'color': '#2ca02c'}
    }
    
    for state, points in centroids.items():
        if points:
            pts = np.array(points)
            cx, cy = np.median(pts[:, 0]), np.mean(pts[:, 1])
            
            ax.annotate(
                state_meta[state]['name'], 
                xy=(cx, cy), 
                xytext=(x_max + 0.2, cy) if state != 'TAM' else (x_max + 0.2, cy-1.5),
                textcoords='data',
                fontsize=12,
                color='#333333',
                va='center',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", color=state_meta[state]['color'], linestyle="--", shrinkA=10, shrinkB=0)
            )

    ax.set_title('Semantic Space', fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=12, colors='#555555')
    ax.set_xlabel('UMAP Dimension 1', fontsize=13, labelpad=10)
    ax.set_ylabel('UMAP Dimension 2', fontsize=13, labelpad=10)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7)
    ax.set_ylim(-7, 12)
    ax.set_xlim(ax.get_xlim()[0], x_max + 2.5)

    handles_models = [
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='gray', markeredgecolor='none', markersize=10, label='GPT-OSS'),
        mlines.Line2D([], [], color='none', marker='s', markerfacecolor='gray', markeredgecolor='none', markersize=9, label='DeepSeek'),
        mlines.Line2D([], [], color='none', marker='*', markerfacecolor='gray', markeredgecolor='k', markersize=14, label='Gemini'),
        mlines.Line2D([], [], color='none', marker='D', markerfacecolor='gray', markeredgecolor='k', markersize=9, label='GPT-5'),
        mlines.Line2D([], [], color='none', marker='X', markerfacecolor='gray', markeredgecolor='k', markersize=10, label='Claude'),
        mlines.Line2D([], [], color='none', marker='P', markerfacecolor='#FFD700', markeredgecolor='k', markersize=15, label='Ground Truth')
    ]
    
    handles_conds = [
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#1f77b4', markersize=11, label='T (Title Only)'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#ff7f0e', markersize=11, label='TA (Title + Abstract)'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#2ca02c', markersize=11, label='TAM (Full Text)')
    ]

    leg1 = ax.legend(handles=handles_models, loc='upper left', bbox_to_anchor=(0.01, -0.12), ncol=6, frameon=False, title="Marker Shape = Model", title_fontsize=12, fontsize=11)
    ax.add_artist(leg1) 
    ax.legend(handles=handles_conds, loc='upper left', bbox_to_anchor=(0.07, -0.25), ncol=3, frameon=False, title="Marker Color = Information State", title_fontsize=12, fontsize=11)

    plt.subplots_adjust(bottom=0.25, right=0.85)
    plt.savefig('unified_model_cluster_publication.pdf', bbox_inches='tight') 
    plt.show()

def main():
    algor_names = []
    source_labels = []

    # 1. Load Ground Truth First
    gt_path = 'algorithm_ground_truth.txt'
    if os.path.exists(gt_path):
        algor_names.append(gt_path)
        source_labels.append('Ground_Truth')
    else:
        print(f"Warning: {gt_path} not found.")

    # 2. Load BOTH Local Models
    local_models = ['gpt-oss', 'deepseek'] 
    for l_model in local_models:
        for state, folder in [('T', 'algorithm_pseudocode_t'), ('TA', 'algorithm_pseudocode_ta'), ('TAM', 'algorithm_pseudocode_tam')]:
            files = sorted(glob.glob(f'{l_model}/{folder}/*txt'))
            for f in files:
                algor_names.append(f)
                source_labels.append(f'{l_model}_{state}')
    
    # 3. Setup paths for API Models
    api_map = {
        'Gemini': {'T': 'gemini_T_algo.txt', 'TA': 'gemini_TA_algo.txt', 'TAM': 'gemini_TAM_algo.txt'},
        'GPT-5':  {'T': 'gpt_T_algo.txt',    'TA': 'gpt_TA_algo.txt',    'TAM': 'gpt_TAM_algo.txt'},
        'Claude': {'T': 'claude_T_algo.txt', 'TA': 'claude_TA_algo.txt', 'TAM': 'claude_TAM_algo.txt'} 
    }

    # 4. Append API Models data
    for api_name, states_map in api_map.items():
        for state in ['T', 'TA', 'TAM']:
            fpath = os.path.join('api_results', states_map[state])
            if os.path.exists(fpath):
                algor_names.append(fpath)
                source_labels.append(f'{api_name}_{state}')
    
    # 5. Load standard text
    algorithms = [standardize_algorithm_text(open(i, 'r').read()) for i in algor_names]
    
    if not algorithms:
        print("No text files found.")
        return
        
    visualize_unified_semantic_space(algorithms, source_labels)

if __name__ == "__main__":
    main()