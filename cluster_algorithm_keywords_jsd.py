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
    
    # --- Machine Learning & Clustering ---
    "pca", "dbscan", "kmeans", "random forest", 
    "svm", "support vector", "neural network", "xgboost", "gradient boosting",
    "unsupervised", "supervised", "classifier", "encoder", "decoder", 
    "manifold", "generative", "data augmentation", "autoencoders",
    "knn", "gan", "diffusion", "t-sne", 'GBM'
    
    # --- Optimization & Statistics ---
    "least squares", "curve fit", "chi square", "levenberg", "marquardt",
    "mcmc", "markov chain", "monte carlo", "gp", "gaussian process", 
    "kde", "bayesian", "emcee", "prior", "posterior",
    "likelihood", "latent", "forward modeling", "inverse problem",
    "linalg", "variational inference", "influence"
    
]
def clean_text(text):
    text = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).lower()

def get_probability_distribution(directory_path, vectorizer):
    """
    Reads all generated algorithms in a directory and calculates the 
    normalized probability distribution of the whitelist terms.
    """
    file_paths = glob.glob(f"{directory_path}/*txt")
    if not file_paths:
        print(f"WARNING: No files found in {directory_path}")
        return np.zeros(len(vectorizer.vocabulary))
        
    algorithms = [clean_text(open(i, 'r').read()) for i in file_paths]
    
    # 1. Count raw occurrences of whitelist words across ALL algorithms in this state
    count_matrix = vectorizer.fit_transform(algorithms)
    total_counts = np.sum(count_matrix.toarray(), axis=0)
    
    # 2. Normalize into a probability distribution (summing to 1.0)
    # Add a tiny epsilon (1e-10) to avoid absolute zero probabilities
    prob_dist = total_counts + 1e-10 
    prob_dist = prob_dist / np.sum(prob_dist)
    
    return prob_dist

def get_ground_truth_distribution(gt_file_path, vectorizer):
    """
    Reads the single ground truth algorithm and creates its probability vector.
    """
    with open(gt_file_path, 'r') as f:
        gt_text = clean_text(f.read())
    
    # Vectorize the single GT document
    count_matrix = vectorizer.transform([gt_text])
    total_counts = np.sum(count_matrix.toarray(), axis=0)
    
    # Normalize with epsilon to avoid log(0)
    prob_dist = total_counts + 1e-10
    prob_dist = prob_dist / np.sum(prob_dist)
    
    return prob_dist


def main():
    print("Calculating Jensen-Shannon Divergence across information states...\n")
    
    # We use ngram_range=(1,2) to catch two-word phrases like "random forest"
    vectorizer = CountVectorizer(vocabulary=METHODOLOGY_WHITELIST, ngram_range=(1, 2))
    
    # Extract probability distributions for T, TA, and TAM
    # Adjust these folder names to match your exact local directories
    P_T   = get_probability_distribution('algorithm_pseudocode_t', vectorizer)
    P_TA  = get_probability_distribution('algorithm_pseudocode_ta', vectorizer)
    P_TAM = get_probability_distribution('algorithm_pseudocode_tam', vectorizer)


    # 2. Get the Ground Truth distribution
    P_GT = get_ground_truth_distribution('algorithm_ground_turth.txt', vectorizer)

    # 3. Calculate the Final Alignment (The Reproducibility Gap)
    
    
    # --- CALCULATE JENSEN-SHANNON DIVERGENCE ---
    # Note: scipy's jensenshannon returns the distance (square root of divergence). 
    # We square it to get the actual JSD, and set base=2 to measure in Bits.
    
    jsd_T_to_TA  = (jensenshannon(P_T, P_TA, base=2) ** 2)
    jsd_TA_to_TAM = (jensenshannon(P_TA, P_TAM, base=2) ** 2)
    jsd_T_to_TAM  = (jensenshannon(P_T, P_TAM, base=2) ** 2)
    jsd_TAM_to_GT = (jensenshannon(P_TAM, P_GT, base=2) ** 2)

    
    
    print("-" * 50)
    print(f"INFORMATION GAIN (JSD in Bits)")
    print("-" * 50)
    print(f"1. Abstract Gain (T -> TA):       {jsd_T_to_TA:.4f} bits")
    print(f"2. Method Gain   (TA -> TAM):     {jsd_TA_to_TAM:.4f} bits")
    print(f"3. Total Shift   (T -> TAM):      {jsd_T_to_TAM:.4f} bits")
    print(f"Residual Ambiguity (TAM -> Ground Truth): {jsd_TAM_to_GT:.4f} bits")
    print("-" * 50)

    # --- VISUALIZATION: The "Nature Astronomy" Money Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Information Gain Step Chart
    states = ['Prior\n(Title Only)', 'Intermediate\n(Title + Abstract)', 'Posterior\n(Full Text)']
    divergences = [0, jsd_T_to_TA, jsd_T_to_TAM] # Cumulative shift from the Prior
    
    ax1.plot(states, divergences, marker='o', markersize=12, linewidth=3, color='#d62728')
    ax1.fill_between(states, divergences, alpha=0.1, color='#d62728')
    ax1.set_title('Cumulative Algorithmic Shift (Information Gain)', fontsize=14)
    ax1.set_ylabel('Jensen-Shannon Divergence (Bits)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate the jumps
    ax1.annotate(f'+{jsd_T_to_TA:.3f} bits', xy=(0.5, divergences[1]/2), 
                 ha='center', va='center', bbox=dict(boxstyle="round", fc="w", ec="gray"))
    ax1.annotate(f'+{jsd_TA_to_TAM:.3f} bits', xy=(1.5, divergences[1] + (divergences[2]-divergences[1])/2), 
                 ha='center', va='center', bbox=dict(boxstyle="round", fc="w", ec="gray"))

    # Plot 2: Top 8 Vocabulary Shifts
    # Find the words that changed the most between Title-Only and Full Text
    shift_magnitude = np.abs(P_TAM - P_T)
    top_indices = np.argsort(shift_magnitude)[-12:]
    top_words = [METHODOLOGY_WHITELIST[i] for i in top_indices]
    
    y_pos = np.arange(len(top_words))
    height = 0.25
    
    ax2.barh(y_pos - height, P_T[top_indices], height, label='Title Only (Prior)', color='#1f77b4')
    ax2.barh(y_pos, P_TA[top_indices], height, label='Title + Abstract', color='#ff7f0e')
    ax2.barh(y_pos + height, P_TAM[top_indices], height, label='Full Text (Posterior)', color='#2ca02c')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_words)
    ax2.set_title('Probability Distribution of Key Methodologies', fontsize=14)
    ax2.set_xlabel('Probability Mass', fontsize=12)
    ax2.legend()
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()