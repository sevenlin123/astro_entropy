import numpy as np
from sklearn.linear_model import Lasso

# 1. Define a simple spectral model (sine basis functions)
def build_spectral_basis(n_points, n_bases):
    x = np.linspace(0, 1, n_points)
    basis = [np.sin((k + 1) * np.pi * x) for k in range(n_bases)]
    return np.vstack(basis).T  # shape: (n_points, n_bases)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, n_points, n_bases):
    basis = build_synthetic_basis(n_points, n_bases)
    coeffs = np.random.randn(n_samples, n_bases)
    spectra = coeffs @ basis.T
    return spectra, coeffs

def build_synthetic_basis(n_points, n_bases):
    x = np.linspace(0, 1, n_points)
    basis = [np.cos((i + 1) * np.pi * x) for i in range(n_bax? 1?