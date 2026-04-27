import glob
import ast
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# We ignore these base packages because they do not define the core algorithmic 
# methodology (e.g., MCMC, PCA, KDE). They are just data plumbing.
NOISE_MODULES = {
    "numpy", "pandas", "matplotlib", "matplotlib.pyplot", "seaborn", 
    "os", "sys", "time", "math", "datetime", "typing", "warnings", 
    "argparse", "pathlib", "json", "csv", "re"
}


class ImportExtractor(ast.NodeVisitor):
    """Traverses the AST and extracts high-signal methodological imports."""
    def __init__(self):
        self.imports = set()

    def _is_noise(self, module_name):
        """Checks if the module is in our blacklist."""
        if not module_name: 
            return False
        # If the import is "matplotlib.pyplot", base_module is "matplotlib"
        base_module = module_name.split('.')[0]
        return base_module in NOISE_MODULES or module_name in NOISE_MODULES

    def visit_Import(self, node):
        for alias in node.names:
            if not self._is_noise(alias.name):
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module if node.module else ""
        if not self._is_noise(module):
            for alias in node.names:
                # Captures exact specific imports, e.g., "sklearn.decomposition.PCA"
                # or "scipy.stats.gaussian_kde"
                full_import = f"{module}.{alias.name}" if module else alias.name
                self.imports.add(full_import)
        self.generic_visit(node)

def extract_code_imports(source_code):
    try:
        tree = ast.parse(source_code)
        extractor = ImportExtractor()
        extractor.visit(tree)
        if not extractor.imports:
            return "no_methodological_imports"
        return " ".join(sorted(extractor.imports))
    except SyntaxError:
        return "syntax_error"

def cluster_and_visualize_imports(code_list, code_filenames, distance_threshold=0.30):
    print("Parsing codes and extracting high-signal import signatures...")
    
    # 1. Extract imports and separate valid codes from syntax errors
    raw_import_strings = [extract_code_imports(code) for code in code_list]
    
    valid_indices = []
    valid_import_strings = []
    syntax_error_files = []
    
    for i, imp_str in enumerate(raw_import_strings):
        if imp_str == "syntax_error":
            syntax_error_files.append(code_filenames[i])
        else:
            valid_indices.append(i)
            valid_import_strings.append(imp_str)
            
    # --- REPORTING THE VALIDITY RATE ---
    total_codes = len(code_list)
    valid_codes_count = len(valid_indices)
    error_count = len(syntax_error_files)
    validity_rate = (valid_codes_count / total_codes) * 100
    
    print("\n" + "="*40)
    print(f"SYNTAX DIAGNOSTIC REPORT")
    print(f"Total Codes Generated: {total_codes}")
    print(f"Valid Python Scripts:  {valid_codes_count} ({validity_rate:.1f}%)")
    print(f"Syntax Errors Caught:  {error_count}")
    print("="*40 + "\n")
    
    # If everything failed, abort clustering gracefully
    if valid_codes_count == 0:
        print("ERROR: All generated codes contain syntax errors. Cannot perform clustering.")
        return [], 0, 0
        
    # 2. Vectorize ONLY the valid codes
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1)
    
    try:
        feature_matrix = vectorizer.fit_transform(valid_import_strings)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        print("WARNING: All valid scripts were identical or only contained noise imports.")
        feature_matrix = TfidfVectorizer().fit_transform(["fallback"] * valid_codes_count)
        feature_names = ["none"]

    # 3. Calculate Distances & Cluster (on valid codes only)
    distances = cosine_distances(feature_matrix)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric="precomputed")
    valid_labels = clustering.fit_predict(distances)
    
    # 4. Calculate H(C|A) based ONLY on valid, competing implementations
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    probabilities = counts / valid_codes_count
    entropy_H_C_A = -np.sum(probabilities * np.log2(probabilities))
    
    # --- VISUALIZATION BLOCK ---
    if valid_codes_count > 1 and feature_matrix.shape[1] >= 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Bar Chart of Top Packages
        total_scores = np.sum(feature_matrix.toarray(), axis=0)
        num_top_features = min(10, len(feature_names))
        top_indices = np.argsort(total_scores)[-num_top_features:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = total_scores[top_indices]
        
        ax1.barh(top_features, top_scores, color='#1f77b4')
        ax1.set_xlabel('Cumulative TF-IDF Score')
        ax1.set_title('Top Methodological Packages (Valid Codes Only)')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)

        # Plot 2: PCA Scatter Plot
        pca = PCA(n_components=2)
        reduced_matrix = pca.fit_transform(feature_matrix.toarray())
        
        scatter = ax2.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 
                              c=valid_labels, cmap='tab20', s=150, edgecolor='k', zorder=5)
        
        for i, original_idx in enumerate(valid_indices):
            if i < 30: # Limit annotations so it's readable
                txt = original_idx + 1
                ax2.annotate(f"C{txt}", (reduced_matrix[i, 0], reduced_matrix[i, 1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax2.set_title(f'Methodology Space (eps = {distance_threshold})')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Plot 3: Heatmap
        im = ax3.imshow(distances, cmap='viridis_r', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax3, label='Cosine Distance')
        
        ticks = np.arange(valid_codes_count)
        if valid_codes_count <= 30:
            ax3.set_xticks(ticks)
            ax3.set_yticks(ticks)
            ax3.set_xticklabels([f"C{valid_indices[i]+1}" for i in ticks], fontsize=8, rotation=90)
            ax3.set_yticklabels([f"C{valid_indices[i]+1}" for i in ticks], fontsize=8)
            
        ax3.set_title('Pairwise Import Distances')
        plt.suptitle(f'Implementation Variance $H(C|A)$ = {entropy_H_C_A:.3f} bits | Valid Codes: {valid_codes_count}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return valid_labels, entropy_H_C_A, len(unique_labels), valid_indices

def main():
    code_tam_names = glob.glob('codes_TAM/*py')
    code_ta_names = glob.glob('codes_TA/*py')
    code_t_names = glob.glob('codes_T/*py')
    
    # Fixed the typo here to properly include code_t_names
    code_names = code_tam_names + code_ta_names + code_t_names
    code_names = code_tam_names
    
    # Sort files to ensure C1, C2 match the exact same files every time
    code_names = sorted(code_names)
    
    if not code_names:
        print("No Python files found. Check your directory paths.")
        return

    codes = [open(i, 'r').read() for i in code_names]

    valid_labels, entropy, num_clusters, valid_indices = cluster_and_visualize_imports(codes, code_names, distance_threshold=0.30)

    print(f"\nDistinct structural clusters found: {num_clusters}")
    for i, label in enumerate(valid_labels):
        original_file = code_names[valid_indices[i]].split('/')[-1] if '/' in code_names[valid_indices[i]] else code_names[valid_indices[i]].split('\\')[-1]
        print(f"Code {valid_indices[i]+1:02d} | Cluster: {label} | File: {original_file}")



if __name__ == "__main__":
    main()