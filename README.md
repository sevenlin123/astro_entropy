# Astro-Entropy: Probabilistic Spectral Reconstruction for TNOs

A computational framework for the probabilistic reconstruction of full spectra from sparse photometric measurements of Trans-Neptunian Objects (TNOs). This project uses latent-space modeling (PCA) and Bayesian inference to bridge the gap between photometry and spectroscopy.

## Project Overview

This repository contains tools to:
- **Extract** computational algorithms from research papers using LLMs (DeepSeek-R1).
- **Generate** Python implementations based on extracted algorithms.
- **Analyze** the semantic space of generated algorithms using UMAP and Sentence Transformers.
- **Estimate** entropy and information gain in the reconstruction process.

## Repository Structure

```text
astro_entropy/
├── src/                        # Python source code
│   ├── clustering/             # Algorithm analysis and semantic clustering (UMAP/HDBSCAN)
│   ├── generation/             # LLM-based algorithm extraction and code generation
│   ├── analysis/               # Entropy estimation and information theory metrics
│   └── visualization/          # Plotting and result visualization
├── data/                       # Dataset and generated artifacts
│   ├── algorithms/             # Extracted algorithm steps (LLM outputs)
│   ├── codes/                  # Generated Python implementations
│   ├── deepseek/               # Specific outputs from DeepSeek-R1 models
│   └── ground_truth/           # Paper text and reference algorithms
├── prompts/                    # System and few-shot prompts for LLM stages
├── paper/                      # Source paper descriptions and drafts
└── scripts/                    # Utility scripts for batch processing
```

## Key Components

### 1. Generation (`src/generation/`)
- `generate_codes.py`: Uses `DeepSeek-R1-Distill-Qwen-14B` to extract algorithms from paper text and generate corresponding Python code.
- `generate_codes_api.py`: Interface for generating code via LLM APIs.

### 2. Clustering & Analysis (`src/clustering/`)
- `cluster_algorithm.py`: Visualizes the semantic manifold of generated algorithms using `jina-embeddings-v2-base-code` and UMAP to compare how different information states (T, TA, TAM) affect model output.
- `cluster_codes.py`: Performs similar clustering analysis on the generated Python scripts.

### 3. Information Theory (`src/analysis/`)
- `estimate_entropy.py`: Metrics to quantify the information content and entropy of the reconstruction framework.

## Requirements

- Python 3.9+
- `torch`, `transformers` (for local LLM execution)
- `sentence-transformers` (for embeddings)
- `umap-learn`, `scikit-learn`, `hdbscan` (for analysis)
- `matplotlib`, `numpy`, `scipy`

## Usage

1. **Algorithm Extraction**: Run `python src/generation/generate_codes.py` to process the paper text in `data/ground_truth/`.
2. **Analysis**: Use `python src/clustering/cluster_algorithm.py` to generate semantic maps of the extracted methods.
3. **Visualization**: Visualization outputs are typically saved as `.eps` or `.pdf` files in the root or specific data folders.

## Citation

> *Probabilistic Spectral Reconstruction of Trans-Neptunian Objects from Sparse Photometry: A Framework for Taxonomy, Survey Optimization, and Outlier Detection*
