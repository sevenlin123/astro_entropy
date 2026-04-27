# Astro-Entropy: Quantifying Information in Astrophysics Research

A project to measure the information content and "entropy" of a research paper by reconstructing its computational core using Large Language Models (LLMs).

## Project Goal

The primary objective of this project is to apply Information Theory to a specific astrophysics paper: **"Probabilistic Spectral Reconstruction of Trans-Neptunian Objects from Sparse Photometry"**.

By using LLMs (DeepSeek-R1) to extract algorithms and generate Python implementations across different information states, we quantify how much information is required to successfully reproduce the paper's computational framework.

## Methodology

1. **Algorithm Extraction**: Using LLMs to distill the core computational steps from the paper text.
2. **Code Reconstruction**: Generating functional Python code from the extracted algorithms.
3. **Entropy Measurement**: Measuring the "Astro-Entropy"—the uncertainty and information gain involved in the reconstruction process across different levels of prior knowledge.
4. **Semantic Analysis**: Using UMAP and embeddings to visualize the semantic stability and convergence of the generated algorithms.

## Repository Structure

```text
astro_entropy/
├── src/                        # Code for the reconstruction experiment
│   ├── clustering/             # Semantic analysis and clustering of generated outputs
│   ├── generation/             # LLM pipelines for algorithm extraction and code generation
│   ├── analysis/               # Information theory and entropy estimation metrics
│   └── visualization/          # Tools for plotting semantic manifolds and results
├── data/                       # Experimental data
│   ├── ground_truth/           # The source paper text and reference data
│   ├── algorithms/             # LLM-extracted algorithms (experimental samples)
│   ├── codes/                  # LLM-generated Python implementations
│   └── deepseek/               # Specific results using the DeepSeek-R1 model
├── prompts/                    # Prompts used to control the LLM during reconstruction
└── paper/                      # Research context and drafts
```

## Citation

This research measures the information content of:
> *Probabilistic Spectral Reconstruction of Trans-Neptunian Objects from Sparse Photometry: A Framework for Taxonomy, Survey Optimization, and Outlier Detection*
