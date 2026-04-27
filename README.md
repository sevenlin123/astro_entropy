# Astro-Entropy: Semantic Information Metrics for Paper Reconstruction

A research project quantifying the information content of astrophysics literature by measuring the entropy and divergence of computational reconstructions performed by Large Language Models (LLMs).

## Research Framework

This project applies Information Theory to evaluate the "reproducibility density" of a target astrophysics paper: *"Probabilistic Spectral Reconstruction of Trans-Neptunian Objects from Sparse Photometry"*.

We test how the semantic uncertainty (entropy) of a reconstructed algorithm decreases as the LLM is provided with increasing levels of prior information (Information States).

### Information States (Conditions)
- **Condition T (Title Only)**: High entropy state; model must speculate core logic from the title.
- **Condition TA (Title + Abstract)**: Intermediate state; core objectives and methods are defined.
- **Condition TAM (Full Text)**: Low entropy state; full technical details are available for reconstruction.
- **Condition TAMU**: (Title + Abstract + Methods) [Experimental]

## Quantitative Metrics

The project calculates several semantic metrics using `jina-embeddings-v2-base-code` and UMAP dimensionality reduction:
- **Semantic Entropy {emb}(Y|X)*: Measures the uncertainty/diversity of algorithm generations within a single condition.
- **Jensen-Shannon Divergence (JSD)**: Quantifies the semantic shift/information gain between states (e.g.,  \rightarrow TA$).
- **Cosine Distance to Ground Truth**: Measures the alignment accuracy between LLM reconstructions and the actual paper logic.

## Repository Structure

```text
astro_entropy/
├── src/
│   ├── generation/             # LLM pipelines (DeepSeek-R1) for algorithm & code extraction
│   └── clustering/             # Information theory metrics, Bootstrapping, and UMAP visualization
├── data/
│   ├── ground_truth/           # The reference algorithm for comparison
│   ├── deepseek/               # Reconstructions from DeepSeek-R1 Distill Qwen 14B
│   └── gpt-oss/                # Reconstructions from Open-Source GPT models
├── prompts/                    # The target paper text segmented by Information State
└── scripts/                    # Utility scripts for data processing
```

## Key Scripts

- `src/generation/generate_codes.py`: Automated extraction loop using DeepSeek-R1.
- `src/clustering/cluster_algorithm_full.py`: Core analysis engine calculating Entropy, JSD, and generating Unified Semantic Space plots.

## Requirements

- `transformers`, `sentence-transformers`
- `umap-learn`, `scikit-learn`
- `matplotlib`, `numpy`, `scipy`
