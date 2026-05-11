import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# NEW IMPORTS FOR INFORMATION THEORY METRICS
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, cosine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

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
    'pca', 'kde', 'components', 'component', 'eigenvalues', 'eigenvectors',
    "linear regression", "regression", "linear", "logistic regression", "logistic",
    "sampling", "importance sampling", "rejection sampling", "metropolis", "hastings",
    "copula"
]

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
    
    js_distance = jensenshannon(p, q, base=2)
    js_divergence = js_distance ** 2
    
    return js_divergence

def calc_cosine_distance(p, q):
    """Calculates Cosine Distance (1 - Cosine Similarity)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if np.sum(p) == 0 or np.sum(q) == 0:
        return np.nan
        
    return cosine(p, q)

# ==========================================
# BOOTSTRAPPING FUNCTIONS
# ==========================================
def bootstrap_entropy(texts, vectorizer, n_iterations=1000):
    """Calculates 95% CI for Shannon Entropy via bootstrapping texts."""
    if not texts or len(texts) == 1:
        # Cannot bootstrap a single file; return point estimate
        p = get_probability_from_texts(texts, vectorizer)
        val = calc_entropy(p)
        return val, val, val
    
    results = []
    n_docs = len(texts)
    for _ in range(n_iterations):
        idx = np.random.choice(n_docs, size=n_docs, replace=True)
        resampled_texts = [texts[i] for i in idx]
        p = get_probability_from_texts(resampled_texts, vectorizer)
        results.append(calc_entropy(p))
        
    return np.mean(results), np.percentile(results, 2.5), np.percentile(results, 97.5)

def bootstrap_jsd(texts_p, texts_q, vectorizer, n_iterations=1000):
    """Calculates 95% CI for JSD via bootstrapping texts."""
    if not texts_p: return np.nan, np.nan, np.nan
    
    n_p = len(texts_p)
    is_q_fixed = isinstance(texts_q, np.ndarray) # True if comparing vs fixed Ground Truth
    n_q = 0 if is_q_fixed else len(texts_q)
    
    if n_p == 1 and (is_q_fixed or n_q == 1):
        p = get_probability_from_texts(texts_p, vectorizer)
        q = texts_q if is_q_fixed else get_probability_from_texts(texts_q, vectorizer)
        val = calc_jsd(p, q)
        return val, val, val
        
    results = []
    for _ in range(n_iterations):
        # Resample P
        idx_p = np.random.choice(n_p, size=n_p, replace=True)
        resampled_p = [texts_p[i] for i in idx_p]
        p_boot = get_probability_from_texts(resampled_p, vectorizer)
        
        # Resample Q (if it's an ensemble and not a fixed vector)
        if is_q_fixed:
            q_boot = texts_q
        else:
            idx_q = np.random.choice(n_q, size=n_q, replace=True)
            resampled_q = [texts_q[i] for i in idx_q]
            q_boot = get_probability_from_texts(resampled_q, vectorizer)
            
        results.append(calc_jsd(p_boot, q_boot))
        
    return np.mean(results), np.percentile(results, 2.5), np.percentile(results, 97.5)

# ==========================================
# TEXT PROCESSING & CLUSTERING
# ==========================================
def clean_text(text):
    text = re.sub(r'(?i)(?:Step\s*\d+:?|\d+[\.\)]\s*|[-*()])', ' ', text)
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip().lower()

def get_probability_from_texts(texts, vectorizer):
    """Calculates normalized probability distribution for a list of texts."""
    if not texts:
        return np.zeros(len(vectorizer.vocabulary_))
    
    count_matrix = vectorizer.transform(texts)
    total_counts = np.sum(count_matrix.toarray(), axis=0)
    
    if np.sum(total_counts) == 0:
        return np.zeros(len(vectorizer.vocabulary_))
        
    return total_counts / np.sum(total_counts)

def main():
    print("Loading text corpus...")
    
    # 1. LOAD ALL DATA INTO MEMORY
    corpus_dict = {}
    source_labels = []
    all_texts = []

    # Ground Truth Data
    gt_gpt_path = DATA_DIR / 'ground_truth' / 'algorithm_ground_truth.txt'
    gt_gpt_text = clean_text(gt_gpt_path.read_text()) if gt_gpt_path.exists() else ""
    
    if gt_gpt_text:
       all_texts.append(gt_gpt_text)
       source_labels.append('Ground_Truth')

    # Load All 5 Models
    models_to_load = ['gpt-oss', 'deepseek', 'Gemini', 'GPT-5', 'Claude']
    for model in models_to_load:
        corpus_dict[model] = {'T': [], 'TA': [], 'TAM': []}
        for state in ['T', 'TA', 'TAM']:
            if model in ['gpt-oss', 'deepseek']:
                paths = sorted((DATA_DIR / model / f'algorithm_pseudocode_{state.lower()}').glob('*.txt'))
            else:
                api_map = {'Gemini': 'gemini', 'GPT-5': 'gpt', 'Claude': 'claude'}
                prefix = api_map[model]
                paths = [DATA_DIR / 'api_results' / f'{prefix}_{state}_algo.txt']
                
            for p in paths:
                if p.exists():
                    txt = clean_text(p.read_text())
                    corpus_dict[model][state].append(txt)
                    all_texts.append(txt)
                    source_labels.append(f'{model}_{state}')

    if not all_texts:
        print("ERROR: No texts loaded. Check paths.")
        return

    # =========================================================
    # VISUALIZATION SETUP: 2x3 Master Grid
    # =========================================================
    fig = plt.figure(figsize=(18, 14)) 
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], wspace=0.0, hspace=0.45)
    
    ax_pca = fig.add_subplot(gs[0, :])
    ax_t   = fig.add_subplot(gs[1, 0])
    ax_ta  = fig.add_subplot(gs[1, 1], sharey=ax_t)
    ax_tam = fig.add_subplot(gs[1, 2], sharey=ax_t)

    # ---------------------------------------------------------
    # PANEL A: PCA Scatter Plot 
    # ---------------------------------------------------------
    print("Computing PCA (TF-IDF)...")
    tfidf_vectorizer = TfidfVectorizer(vocabulary=METHODOLOGY_WHITELIST, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    if np.sum(tfidf_matrix.toarray()) > 0:
        pca = PCA(n_components=2, random_state=42) 
        reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
        
        ax_pca.set_facecolor('#FAFAFA')
        ax_pca.spines['top'].set_color('#e0e0e0')
        ax_pca.spines['right'].set_color('#e0e0e0')
        ax_pca.spines['bottom'].set_color('#cccccc')
        ax_pca.spines['left'].set_color('#cccccc')

        styles = {
            'Ground_Truth': {'c': '#FFD700', 'm': 'P', 's': 500, 'z': 20},
            'gpt-oss_T':  {'c': '#1f77b4', 'm': 'o', 's': 60, 'z': 5},
            'deepseek_T': {'c': '#1f77b4', 'm': 's', 's': 60, 'z': 5},
            'Gemini_T':   {'c': '#1f77b4', 'm': '*', 's': 300, 'z': 10},
            'GPT-5_T':    {'c': '#1f77b4', 'm': 'D', 's': 120, 'z': 10},
            'Claude_T':   {'c': '#1f77b4', 'm': 'X', 's': 150, 'z': 10},
            'gpt-oss_TA':  {'c': '#ff7f0e', 'm': 'o', 's': 60, 'z': 5},
            'deepseek_TA': {'c': '#ff7f0e', 'm': 's', 's': 60, 'z': 5},
            'Gemini_TA':   {'c': '#ff7f0e', 'm': '*', 's': 300, 'z': 10},
            'GPT-5_TA':    {'c': '#ff7f0e', 'm': 'D', 's': 120, 'z': 10},
            'Claude_TA':   {'c': '#ff7f0e', 'm': 'X', 's': 150, 'z': 10},
            'gpt-oss_TAM':  {'c': '#2ca02c', 'm': 'o', 's': 60, 'z': 5},
            'deepseek_TAM': {'c': '#2ca02c', 'm': 's', 's': 60, 'z': 5},
            'Gemini_TAM':   {'c': '#2ca02c', 'm': '*', 's': 300, 'z': 10},
            'GPT-5_TAM':    {'c': '#2ca02c', 'm': 'D', 's': 120, 'z': 10},
            'Claude_TAM':   {'c': '#2ca02c', 'm': 'X', 's': 150, 'z': 10}
        }
        
        ordered_keys = ['Ground_Truth'] + [f"{m}_{s}" for s in ['T', 'TA', 'TAM'] for m in models_to_load]
        centroids = {'T': [], 'TA': [], 'TAM': []}

        for slabel in ordered_keys:
            idx = [i for i, src in enumerate(source_labels) if src == slabel]
            if not idx: continue
            
            style = styles.get(slabel, {'c': 'gray', 'm': '.', 's': 50, 'z': 1})
            is_api = style['m'] not in ['o', 's']
            alpha_val = 0.9 if is_api else 0.3
            
            points_x = reduced_matrix[idx, 0]
            points_y = reduced_matrix[idx, 1]
            
            ax_pca.scatter(
                points_x, points_y, 
                c=style['c'], marker=style['m'], s=style['s'], 
                edgecolor='k' if is_api else 'none', linewidth=1.0 if is_api else 0,
                zorder=style['z'], alpha=alpha_val
            )

            if '_T' in slabel and 'TA' not in slabel: centroids['T'].extend(zip(points_x, points_y))
            elif '_TA' in slabel and 'TAM' not in slabel: centroids['TA'].extend(zip(points_x, points_y))
            elif '_TAM' in slabel: centroids['TAM'].extend(zip(points_x, points_y))
            
        x_max = ax_pca.get_xlim()[1]
        state_meta = {
            'T': {'name': 'Condition T\n(Title Only)', 'color': '#1f77b4'},
            'TA': {'name': 'Condition TA\n(Title + Abstract)', 'color': '#ff7f0e'},
            'TAM': {'name': 'Condition TAM\n(Full Text)', 'color': '#2ca02c'}
        }
        
        for state, points in centroids.items():
            if points:
                pts = np.array(points)
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                
                ax_pca.annotate(
                    state_meta[state]['name'], 
                    xy=(cx, cy), 
                    xytext=(x_max + 0.0, cy) if state != 'T' else (x_max + 0.0, cy - 0.1),
                    textcoords='data',
                    fontsize=14,
                    color='#333333',
                    va='center',
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color=state_meta[state]['color'], linestyle="--", shrinkA=10, shrinkB=0)
                )
        ax_pca.set_title('A. Lexical Convergence Space (PCA on TF-IDF Vectors)', fontsize=14, fontweight='bold')
        ax_pca.set_xlabel('Principal Component 1', fontsize=14)
        ax_pca.set_ylabel('Principal Component 2', fontsize=14)
        ax_pca.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7)
        ax_pca.set_xlim(ax_pca.get_xlim()[0], x_max+0.2) 

        # Legends
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

        leg_models = ax_pca.legend(handles=handles_models, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False, title="Marker Shape = Model", title_fontsize=14, fontsize=11)
        ax_pca.add_artist(leg_models) 
        ax_pca.legend(handles=handles_conds, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, title="Marker Color = Information State", title_fontsize=14, fontsize=11)

    # ---------------------------------------------------------
    # PANELS B, C, D: Lexical Wave Collapses
    # ---------------------------------------------------------
    print("Computing Word Probability Distributions (Counts)...")
    count_vectorizer = CountVectorizer(vocabulary=METHODOLOGY_WHITELIST, ngram_range=(1, 2))
    count_vectorizer.fit(all_texts) 
    
    global_mass = np.zeros(len(count_vectorizer.vocabulary_))
    for m in models_to_load:
        for s in ['T', 'TA', 'TAM']:
            global_mass += get_probability_from_texts(corpus_dict[m][s], count_vectorizer)
    
    if gt_gpt_text:
        global_mass += get_probability_from_texts([gt_gpt_text], count_vectorizer)
        gt_probs = get_probability_from_texts([gt_gpt_text], count_vectorizer)
    else:
        gt_probs = np.zeros(len(count_vectorizer.vocabulary_))
        
    top_indices = np.argsort(global_mass)[-10:]
    top_words = [METHODOLOGY_WHITELIST[i] for i in top_indices]
    
    model_colors = {
        'gpt-oss': '#1f77b4',   
        'deepseek': '#ff7f0e',  
        'Gemini': '#2ca02c',    
        'GPT-5': '#d62728',     
        'Claude': '#9467bd'     
    }
    
    width = 0.14
    x = np.arange(len(top_words))
    gt_probs_top = gt_probs[top_indices]

    states_to_plot = [
        ('T', ax_t, 'B. Prior State (T)'),
        ('TA', ax_ta, 'C. Intermediate State (TA)'),
        ('TAM', ax_tam, 'D. Posterior State (TAM)')
    ]

    for state_name, ax, title in states_to_plot:
        for i, model in enumerate(models_to_load):
            offset = (i - 2.5) * width
            model_probs = get_probability_from_texts(corpus_dict[model][state_name], count_vectorizer)[top_indices]
            
            ax.bar(x + offset, model_probs, width, label=f'{model}' if state_name == 'TAM' else "", color=model_colors[model], alpha=0.85)
            
        ax.bar(x + 2.5 * width, gt_probs_top, width, label='Ground Truth' if state_name == 'TAM' else "", color='#FFD700', edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(top_words, rotation=55, ha='right', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if ax == ax_t:
            ax.set_ylabel('Keyword Probability Mass', fontsize=14)
        else:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax_tam.legend(fontsize=14, loc=2, frameon=True, shadow=True)
    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.95)
    
    # =========================================================
    # EXPERIMENT 1: QUANTITATIVE METRICS WITH BOOTSTRAPPED CIs
    # =========================================================
    print("\n" + "="*60)
    print(" EXPERIMENT 1: QUANTITATIVE METRICS (KEYWORD SPACE)")
    print("="*60)

    # Set seed for exactly reproducible confidence intervals in the paper
    np.random.seed(42)

    for model in ['gpt-oss', 'deepseek']:
        print(f"\nModel: {model.upper()}")
        print("-" * 60)
        
        texts_T = corpus_dict[model]['T']
        texts_TA = corpus_dict[model]['TA']
        texts_TAM = corpus_dict[model]['TAM']
        
        # 1. Shannon Entropies
        print("1. Shannon Entropy H(Y|X) (bits) [Mean, 95% CI]:")
        for state, texts in [('T', texts_T), ('TA', texts_TA), ('TAM', texts_TAM)]:
            mean_val, low, high = bootstrap_entropy(texts, count_vectorizer)
            print(f"   {state:<3} : {mean_val:.4f} [{low:.4f}, {high:.4f}]")
        print()
        
        # 2. Inter-State Divergence (JSD in bits)
        print("2. Inter-State Divergence (JSD in bits) [Mean, 95% CI]:")
        transitions = [('T -> TA', texts_T, texts_TA), 
                       ('TA -> TAM', texts_TA, texts_TAM), 
                       ('T -> TAM', texts_T, texts_TAM)]
        for label, p_texts, q_texts in transitions:
            mean_val, low, high = bootstrap_jsd(p_texts, q_texts, count_vectorizer)
            print(f"   {label:<9}: {mean_val:.4f} [{low:.4f}, {high:.4f}]")
        print()
        
        # 3. Ground-Truth Alignment (JSD vs GT in bits)
        print("3. Ground-Truth Alignment (JSD vs GT in bits) [Mean, 95% CI]:")
        for state, texts in [('T', texts_T), ('TA', texts_TA), ('TAM', texts_TAM)]:
            mean_val, low, high = bootstrap_jsd(texts, gt_probs, count_vectorizer)
            print(f"   {state:<3} vs GT : {mean_val:.4f} [{low:.4f}, {high:.4f}]")
        
    # =========================================================
    # EXPERIMENT 2: API COSINE DISTANCES (LEXICAL SPACE)
    # =========================================================
    print("\n" + "="*60)
    print(" EXPERIMENT 2: API COSINE DISTANCES (LEXICAL SPACE)")
    print("="*60)

    api_models = ['Gemini', 'GPT-5', 'Claude']
    api_states = ['T', 'TA', 'TAM']

    # --- TABLE 6 ---
    print("\nTABLE 6: COSINE DISTANCE TO GROUND TRUTH")
    print(f"{'State':<10} | {'Gemini':<12} | {'GPT-5':<12} | {'Claude':<12}")
    print("-" * 60)

    for state in api_states:
        row_dists = []
        for model in api_models:
            if corpus_dict[model][state]:
                model_probs = get_probability_from_texts(corpus_dict[model][state], count_vectorizer)
                dist = calc_cosine_distance(model_probs, gt_probs)
                row_dists.append(f"{dist:.4f}")
            else:
                row_dists.append("N/A")
        print(f"{state:<10} | {row_dists[0]:<12} | {row_dists[1]:<12} | {row_dists[2]:<12}")

    # --- TABLE 7 ---
    print("\nTABLE 7: INTRA-MODEL TRAJECTORY SHIFTS")
    print(f"{'Model':<10} | {'T -> TA':<12} | {'TA -> TAM':<12} | {'T -> TAM':<12}")
    print("-" * 60)

    for model in api_models:
        vecs = {}
        for state in api_states:
            if corpus_dict[model][state]:
                 vecs[state] = get_probability_from_texts(corpus_dict[model][state], count_vectorizer)
            else:
                 vecs[state] = None

        if all(v is not None for v in vecs.values()):
            shift_T_TA = calc_cosine_distance(vecs['T'], vecs['TA'])
            shift_TA_TAM = calc_cosine_distance(vecs['TA'], vecs['TAM'])
            shift_T_TAM = calc_cosine_distance(vecs['T'], vecs['TAM'])
            print(f"{model:<10} | {shift_T_TA:<12.4f} | {shift_TA_TAM:<12.4f} | {shift_T_TAM:<12.4f}")
        else:
            print(f"{model:<10} | {'Missing Data':<12} | {'Missing Data':<12} | {'Missing Data':<12}")

    print("="*60 + "\n")

    plt.savefig(PROJECT_ROOT / 'lexical_state_grouped_hero.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
