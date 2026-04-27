import re
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon

# ==========================================
# THE WHITELIST: The Only Words That Matter
# ==========================================
# The vectorizer will ignore literally every word in the text EXCEPT these.
METHODOLOGY_WHITELIST = [
    "dbscan", "kmeans", "random forest", 
    "svm", "support vector", "neural network", "xgboost", "gradient boosting",
    "unsupervised", "supervised", 
    "manifold", "generative", "data augmentation", "autoencoders",
    "knn", "gan", "diffusion", "t-sne", 'gbm', 'gbdt',
    "least squares", "curve fit", "chi square", "levenberg", "marquardt",
    "mcmc", "markov chain", "monte carlo", "gp", "gaussian process", 
    "bayesian", "emcee", 'principal component analysis',
    "likelihood", "forward modeling", "inverse problem",
    "linalg", "variational inference", "influence", 'kernel density estimation',
    'pca', 'kde', 'components', 'component', 'eigenvalues', 'eigenvectors',
    "linear regression", "regression", "linear", "logistic regression", "logistic",
    "sampling", "importance sampling", "rejection sampling", "metropolis", "hastings",
    "copula"
]


def standardize_algorithm_text(text):
    """Standardizes text and removes arbitrary formatting."""
    standardized = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    standardized = standardized.replace('\n', ' ')
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized.lower()

def cluster_algorithms_whitelist(cleaned_algorithms, distance_threshold=0.20):
    print("Extracting algorithm signatures using strict Methodology Whitelist...")
    
    # 1. Strict Whitelist Vectorization
    # ngram_range=(1,2) allows it to find "pca" (1 word) AND "least squares" (2 words)
    vectorizer = TfidfVectorizer(vocabulary=METHODOLOGY_WHITELIST, ngram_range=(1, 2))
    feature_matrix = vectorizer.fit_transform(cleaned_algorithms)
    feature_names = vectorizer.get_feature_names_out()
    
    # --- Check for empty vectors (Algorithms with ZERO math words) ---
    # If an algorithm used none of our whitelist words, its sum is 0
    row_sums = np.sum(feature_matrix.toarray(), axis=1)
    if np.all(row_sums == 0):
        print("ERROR: None of the algorithms contained ANY words from the whitelist.")
        return [], 0, 0, 0

    # 2. Calculate Distance & Cluster
    distances = cosine_distances(feature_matrix)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(distances)
    
    # 3. Calculate H(A|P)
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(cleaned_algorithms)
    entropy_H_A_P = -np.sum(probabilities * np.log2(probabilities))
    
    # --- VISUALIZATION BLOCK ---
    if len(cleaned_algorithms) > 1 and feature_matrix.shape[1] >= 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Top Whitelist Hits
        total_scores = np.sum(feature_matrix.toarray(), axis=0)
        # Filter out words that didn't appear at all (score == 0)
        active_indices = np.where(total_scores > 0)[0]
        
        if len(active_indices) > 0:
            top_indices = active_indices[np.argsort(total_scores[active_indices])[-10:]]
            top_words = [feature_names[i] for i in top_indices]
            top_word_scores = total_scores[top_indices]
            
            ax1.barh(top_words, top_word_scores, color='#d62728') 
            ax1.set_title('Detected Methodology (Whitelist Hits)')
            ax1.set_xlabel('Cumulative TF-IDF Score')
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, "No Whitelist Words Found", ha='center')

        # Plot 2: PCA Scatter Plot
        # We need at least 2 features to do 2D PCA. If only 1 whitelist word was found across
        # all documents, PCA will fail. We handle that edge case here.
        if len(active_indices) >= 2:
            pca = PCA(n_components=2)
            reduced_matrix = pca.fit_transform(feature_matrix.toarray())
            
            scatter = ax2.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 
                                  c=labels, cmap='tab20', s=150, edgecolor='k', zorder=5)
            
            for i, txt in enumerate(range(1, len(cleaned_algorithms) + 1)):
                if i < 30:
                    ax2.annotate(f"A{txt}", (reduced_matrix[i, 0], reduced_matrix[i, 1]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Not enough features for 2D PCA", ha='center')
            
        ax2.set_title(f'Methodology Space (eps = {distance_threshold})')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Plot 3: Cosine Distance Heatmap
        im = ax3.imshow(distances, cmap='viridis_r', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax3, label='Cosine Distance')
        
        ticks = np.arange(len(cleaned_algorithms))
        if len(cleaned_algorithms) <= 30:
            ax3.set_xticks(ticks)
            ax3.set_yticks(ticks)
            ax3.set_xticklabels([f"A{i+1}" for i in ticks], fontsize=8, rotation=90)
            ax3.set_yticklabels([f"A{i+1}" for i in ticks], fontsize=8)
            
        ax3.set_title('Pairwise Whitelist Distances')
        plt.suptitle(f'Algorithmic Ambiguity $H(A|P)$ = {entropy_H_A_P:.3f} bits | Clusters: {len(unique_labels)}', fontsize=16)
        plt.tight_layout()
        plt.show()

    return labels, entropy_H_A_P, len(unique_labels)

def main():

    model = 'deepseek'
    model = 'gpt-oss'

    algor_tam_names = glob.glob(f'{model}/algorithm_pseudocode_tam/*txt')
    algor_ta_names = glob.glob(f'{model}/algorithm_pseudocode_ta/*txt')
    algor_t_names = glob.glob(f'{model}/algorithm_pseudocode_t/*txt')
    
    algor_names = algor_t_names # + algor_ta_names + algor_tam_names
    
    if not algor_names:
        print("No text files found. Check your path.")
        return

    algorithms = [standardize_algorithm_text(open(i, 'r').read()) for i in algor_names]
    
    # Note: Whitelist thresholds can often be lower (e.g., 0.20) because the 
    # vectors are so sparse and specific.
    labels, entropy, num_clusters = cluster_algorithms_whitelist(algorithms, distance_threshold=0.1)
    
    print(f"\nDistinct algorithmic clusters found: {num_clusters}")
    for i, label in enumerate(labels):
        filename = algor_names[i].split('/')[-1] if '/' in algor_names[i] else algor_names[i].split('\\')[-1]
        print(f"Algorithm {i+1:02d} | Cluster: {label} | File: {filename}")
        
    print(f"\nAlgorithmic Ambiguity H(A|P): {entropy:.3f} bits")

if __name__ == "__main__":
    main()