import glob
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

# ==========================================
# THE DOMAIN WHITELIST
# ==========================================
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
    'pca', 'kde', 'components', 'component', 'eigenvalues', 'eigenvectors'
]

def clean_text(text):
    """
    Removes arbitrary list formatting and standardizes text for embedding.
    """
    standardized = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|-|\*)', ' ', text)
    standardized = standardized.replace('\n', ' ')
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized.lower()


def get_probability_distribution(directory_path, vectorizer):
    """Calculates the normalized probability distribution of whitelist terms."""
    file_paths = glob.glob(f"{directory_path}/*txt")
    if not file_paths:
        # Return a flat uniform distribution if directory is missing to avoid crashing
        return np.ones(len(vectorizer.vocabulary)) / len(vectorizer.vocabulary)
        
    algorithms = [clean_text(open(i, 'r').read()) for i in file_paths]
    count_matrix = vectorizer.fit_transform(algorithms)
    total_counts = np.sum(count_matrix.toarray(), axis=0)
    
    prob_dist = total_counts + 1e-10 
    return prob_dist / np.sum(prob_dist)

def get_ground_truth_distribution(file_path, vectorizer):
    """Calculates the delta function distribution for the Ground Truth file."""
    try:
        gt_text = clean_text(open(file_path, 'r').read())
    except FileNotFoundError:
        print(f"WARNING: Ground truth file '{file_path}' not found. Using uniform fallback.")
        return np.ones(len(vectorizer.vocabulary)) / len(vectorizer.vocabulary)
        
    count_matrix = vectorizer.transform([gt_text])
    total_counts = np.sum(count_matrix.toarray(), axis=0)
    
    prob_dist = total_counts + 1e-10
    return prob_dist / np.sum(prob_dist)

def main():
    print("Calculating Jensen-Shannon Divergence and Wave Collapse...\n")
    
    vectorizer = CountVectorizer(vocabulary=METHODOLOGY_WHITELIST, ngram_range=(1, 2))
    model = 'deepseek'
    model = 'gpt-oss'
    # 1. Extract distributions
    P_T   = get_probability_distribution(f'{model}/algorithm_pseudocode_t', vectorizer)
    P_TA  = get_probability_distribution(f'{model}/algorithm_pseudocode_ta', vectorizer)
    P_TAM = get_probability_distribution(f'{model}/algorithm_pseudocode_tam', vectorizer)
    
    # Update this to point to your actual ground truth file
    P_GT  = get_ground_truth_distribution('algorithm_ground_truth.txt', vectorizer)
    P_GT  = get_ground_truth_distribution('pseudocode_ground_truth.txt', vectorizer)
    
    # 2. Calculate JSD (Squared to get actual bits, base 2)
    jsd_T_to_TA   = (jensenshannon(P_T, P_TA, base=2) ** 2)
    jsd_TA_to_TAM = (jensenshannon(P_TA, P_TAM, base=2) ** 2)
    jsd_T_to_TAM  = (jensenshannon(P_T, P_TAM, base=2) ** 2)
    jsd_T_to_GT = (jensenshannon(P_T, P_GT, base=2) ** 2)
    jsd_TA_to_GT = (jensenshannon(P_TA, P_GT, base=2) ** 2)
    jsd_TAM_to_GT = (jensenshannon(P_TAM, P_GT, base=2) ** 2)
    
    print("-" * 50)
    print(f"INFORMATION GAIN (JSD in Bits)")
    print("-" * 50)
    print(f"1. Abstract Gain (T -> TA):       {jsd_T_to_TA:.4f} bits")
    print(f"2. Method Gain   (TA -> TAM):     {jsd_TA_to_TAM:.4f} bits")
    print(f"3. Total Shift   (T -> TAM):      {jsd_T_to_TAM:.4f} bits")
    print(f"Residual Ambiguity (T -> GT):   {jsd_T_to_GT:.4f} bits")
    print(f"Residual Ambiguity (TA -> GT):   {jsd_TA_to_GT:.4f} bits")
    print(f"Residual Ambiguity (TAM -> GT):   {jsd_TAM_to_GT:.4f} bits")
    print("-" * 50)

    # --- VISUALIZATION BLOCK ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Cumulative Information Gain (The Trajectory)
    states = ['Prior\n(Title Only)', 'Intermediate\n(Title + Abstract)', 'Posterior\n(Full Text)']
    divergences = [0, jsd_T_to_TA, jsd_T_to_TAM] 
    
    ax1.plot(states, divergences, marker='o', markersize=12, linewidth=3, color='#1f77b4')
    ax1.fill_between(states, divergences, alpha=0.1, color='#1f77b4')
    ax1.set_title('Cumulative Algorithmic Shift', fontsize=14)
    ax1.set_ylabel('Jensen-Shannon Divergence (Bits)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1.annotate(f'+{jsd_T_to_TA:.3f} bits', xy=(0.5, divergences[1]/2), 
                 ha='center', va='center', bbox=dict(boxstyle="round", fc="w", ec="gray"))
    ax1.annotate(f'+{jsd_TA_to_TAM:.3f} bits', xy=(1.5, divergences[1] + (divergences[2]-divergences[1])/2), 
                 ha='center', va='center', bbox=dict(boxstyle="round", fc="w", ec="gray"))

    # Plot 2: Visualizing the Collapse (Grouped Bar Chart)
    # Find the top 6 most relevant words across T, TAM, and GT to plot
    composite_mass = P_T + P_TAM + P_GT
    top_indices = np.argsort(composite_mass)[-10:]
    top_words = [METHODOLOGY_WHITELIST[i] for i in top_indices]
    
    val_T   = P_T[top_indices]
    val_TA   = P_TA[top_indices]
    val_TAM = P_TAM[top_indices]
    val_GT  = P_GT[top_indices]
    
    x = np.arange(len(top_words))
    width = 0.15
    
    ax2.bar(x - 1.5*width, val_T, width, label='Prior (Title Only)', color='#1f77b4', alpha=0.8)
    ax2.bar(x - 0.5* width, val_TA, width, label='Posterior 0 (Title+Abstract)', color='#2ca02c', alpha=0.8)
    ax2.bar(x + 0.5*width, val_TAM, width, label='Posterior 1 (Full Text)', color='#ff7f0e', alpha=0.9)
    # Ground Truth is the Delta Function Spike
    ax2.bar(x + 1.5*width, val_GT, width, label='Ground Truth (Delta)', color='#d62728', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_words, rotation=30, ha='right', fontsize=11)
    ax2.set_title(f'Wave Collapse (Residual Gap: {jsd_TAM_to_GT:.3f} bits)', fontsize=14)
    ax2.set_ylabel('Probability Mass', fontsize=12)
    ax2.set_ylim(0, 0.6) # Max probability is 1.0
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()