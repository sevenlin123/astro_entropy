import glob
import re
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

def standardize_algorithm_text(text):
    """Removes arbitrary formatting to standardize the text for embedding."""
    standardized = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    return re.sub(r'\s+', ' ', standardized.replace('\n', ' ')).lower()

def load_directory(path):
    """Loads and standardizes all text files in a directory."""
    files = sorted(glob.glob(f"{path}/*txt"))
    if not files:
        return []
    return [standardize_algorithm_text(open(f, 'r').read()) for f in files]

def get_probability_distribution(labels, all_unique_labels):
    """Converts a list of cluster labels into a probability distribution."""
    counts = np.array([np.sum(labels == label) for label in all_unique_labels])
    prob_dist = counts + 1e-10
    return prob_dist / np.sum(prob_dist)

def main():
    print("Loading text files...")

    # Choose your model folder
    model = 'deepseek'
    #model = 'gpt-oss'

    texts_T   = load_directory(f'{model}/algorithm_pseudocode_t')
    texts_TA  = load_directory(f'{model}/algorithm_pseudocode_ta')
    texts_TAM = load_directory(f'{model}/algorithm_pseudocode_tam')
    
    try:
        if model == 'deepseek':
            texts_GT = [standardize_algorithm_text(open('algorithm_ground_truth.txt', 'r').read())]
        else:
            texts_GT = [standardize_algorithm_text(open('pseudocode_ground_truth.txt', 'r').read())]
    except FileNotFoundError:
        print("ERROR: 'algorithm_ground_truth.txt' not found.")
        return

    len_T   = len(texts_T)
    len_TA  = len(texts_TA)
    len_TAM = len(texts_TAM)
    len_GT  = len(texts_GT)
    
    if len_T == 0 or len_TAM == 0:
        print("ERROR: Missing text files in directories.")
        return

    all_texts = texts_T + texts_TA + texts_TAM + texts_GT

    print("Generating SentenceTransformer embeddings for the unified corpus...")
    model_embedder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code", 
        trust_remote_code=True,
        device="mps"
    )
    
    model_embedder.max_seq_length = 2048 
    embeddings = model_embedder.encode(
        all_texts, 
        batch_size=8, 
        show_progress_bar=True 
    )

    # ---------------------------------------------------------
    # Direct Semantic Distance to Ground Truth (Trajectory)
    # ---------------------------------------------------------
    print("\n--- GROUND TRUTH ACCURACY TRAJECTORY ---")
    
    # Extract embeddings for each state
    embeddings_T   = embeddings[ : len_T]
    embeddings_TA  = embeddings[len_T : len_T + len_TA]
    embeddings_TAM = embeddings[len_T + len_TA : len_T + len_TA + len_TAM]
    embedding_GT   = embeddings[-len_GT:]
    
    # Calculate distances to GT for each state
    dist_T_to_GT   = cosine_distances(embeddings_T, embedding_GT).flatten()
    dist_TA_to_GT  = cosine_distances(embeddings_TA, embedding_GT).flatten()
    dist_TAM_to_GT = cosine_distances(embeddings_TAM, embedding_GT).flatten()
    
    # Calculate Min and Avg
    min_T, avg_T     = np.min(dist_T_to_GT), np.mean(dist_T_to_GT)
    min_TA, avg_TA   = np.min(dist_TA_to_GT), np.mean(dist_TA_to_GT)
    min_TAM, avg_TAM = np.min(dist_TAM_to_GT), np.mean(dist_TAM_to_GT)
    
    print(f"Prior (Title Only)       -> Min Dist: {min_T:.4f} | Avg Dist: {avg_T:.4f}")
    print(f"Intermediate (Title+Abs) -> Min Dist: {min_TA:.4f} | Avg Dist: {avg_TA:.4f}")
    print(f"Posterior (Full Text)    -> Min Dist: {min_TAM:.4f} | Avg Dist: {avg_TAM:.4f}")
    print("----------------------------------------\n")

    # ---------------------------------------------------------
    # UMAP + HDBSCAN Pipeline (For Entropy)
    # ---------------------------------------------------------
    print("Applying UMAP Dimensionality Reduction...")
    STATIC_NEIGHBORS = 5 
    
    if len(all_texts) <= STATIC_NEIGHBORS:
        raise ValueError(f"Sample size too small for n_neighbors={STATIC_NEIGHBORS}.")
        
    reducer = umap.UMAP(n_neighbors=STATIC_NEIGHBORS, n_components=2, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    print("Clustering unified semantic space with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    all_labels = clusterer.fit_predict(umap_embeddings)

    labels_T   = all_labels[ : len_T]
    labels_TA  = all_labels[len_T : len_T + len_TA]
    labels_TAM = all_labels[len_T + len_TA : len_T + len_TA + len_TAM]
    labels_GT  = all_labels[-len_GT : ]

    unique_labels = np.unique(all_labels)
    P_T   = get_probability_distribution(labels_T, unique_labels)
    P_TA  = get_probability_distribution(labels_TA, unique_labels)
    P_TAM = get_probability_distribution(labels_TAM, unique_labels)
    P_GT  = get_probability_distribution(labels_GT, unique_labels)

# Calculate Jensen-Shannon Divergence
    jsd_T_to_TA   = (jensenshannon(P_T, P_TA, base=2) ** 2)
    jsd_TA_to_TAM = (jensenshannon(P_TA, P_TAM, base=2) ** 2)
    jsd_T_to_TAM  = (jensenshannon(P_T, P_TAM, base=2) ** 2)

    # --- ADD THIS PRINT BLOCK BACK IN ---
    print("\n--- SEMANTIC INFORMATION GAIN (JSD in Bits) ---")
    print(f"1. Abstract Gain (T -> TA):   {jsd_T_to_TA:.4f} bits")
    print(f"2. Method Gain   (TA -> TAM): {jsd_TA_to_TAM:.4f} bits")
    print(f"3. Total Shift   (T -> TAM):  {jsd_T_to_TAM:.4f} bits")
    print("-----------------------------------------------\n")


    # ---------------------------------------------------------
    # Visualization (Now with 3 Subplots!)
    # ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # Plot 1: Cumulative Information Gain
    states = ['T\n(Title)', 'TA\n(Title+Abs)', 'TAM\n(Full Text)']
    divergences = [0, jsd_T_to_TA, jsd_T_to_TAM] 
    
    ax1.plot(states, divergences, marker='o', markersize=10, linewidth=3, color='#9467bd') 
    ax1.fill_between(states, divergences, alpha=0.1, color='#9467bd')
    ax1.set_title('Semantic Information Gain (JSD)', fontsize=14)
    ax1.set_ylabel('Divergence (Bits)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Semantic Wave Collapse
    x = np.arange(len(unique_labels))
    width = 0.25
    
    ax2.bar(x - width, P_T, width, label='Prior (T)', color='#1f77b4', alpha=0.8)
    ax2.bar(x, P_TAM, width, label='Posterior (TAM)', color='#ff7f0e', alpha=0.9)
    ax2.bar(x + width, P_GT, width, label='Ground Truth', color='#d62728', edgecolor='black')
    
    ax2.set_xticks(x)
    x_labels = [f"C{i}" if i != -1 else "Noise" for i in unique_labels]
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_title('Probability Distribution', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot 3: Distance to Ground Truth Trajectory (NEW)
    min_distances = [min_T, min_TA, min_TAM]
    avg_distances = [avg_T, avg_TA, avg_TAM]
    
    ax3.plot(states, avg_distances, marker='s', markersize=8, linewidth=2, color='#d62728', label='Average Distance')
    ax3.plot(states, min_distances, marker='o', markersize=8, linewidth=2, color='#2ca02c', linestyle='--', label='Best Generation (Min)')
    
    ax3.set_title('Distance to Ground Truth', fontsize=14)
    ax3.set_ylabel('Cosine Distance (Lower is Better)', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Add a visual baseline for what is considered a "good" match
    ax3.axhline(y=0.15, color='gray', linestyle=':', alpha=0.8)
    ax3.text(0, 0.16, 'Highly Accurate Match Threshold', color='gray', fontsize=9, va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()