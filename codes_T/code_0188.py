#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge
from scipy.integrate import trapz

# --------------------- Spectral Model ---------------------
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_basis_functions(wavelengths, n_basis, seed=0):
    rng = np.random.default_rng(seed)
    mus = rng.uniform(wavelengths[0], wavelengths[-1], size=n_basis)
    sigma = (wavelengths[-1] - wavelengths[0]) / (4 * n_basis)
    basis = [gaussian(wavelengths, mu, sigma) for mu in mus]
    return np.array(basis)  # shape (n_basis, N)

# --------------------- Synthetic Spectra ---------------------
def synthesize_spectrum(basis, coef, noise_std=0.0, seed=None):
    rng = np.random.default_rng(seed)
    spectrum = np.dot(coef, basis)
    if noise_std > 0.0:
        spectrum += rng.normal(scale=noise_std, size=spectrum.shape)
    return spectrum

# --------------------- Filters ---------------------
def generate_filter(wavelengths, center, width):
    return gaussian(wavelengths, center, width)

def generate_filters(wavelengths, centers, width):
    return [generate_filter(wavelengths, c, width) for c in centers]

# --------------------- Photometry ---------------------
def integrate_band(spectrum, filt):
    num = trapz(spectrum * filt, spectrum= None, axis=-1)
    den = trapz(filt, spectrum=None, axis=-1)
    return num / den

def compute_photometry(spectrum, filters):
    return np.array([integrate_band(spectrum, f) for f in filters])

# --------------------- Reconstruction ---------------------
def build_design_matrix(basis, filters):
    """Matrix A where A_{ij} = <basis_i, filter_j> / <filter_j, 1>"""
    N_b, N_s = basis.shape
    N_f = len(filters)
    A = np.empty((N_f, N_b))
    for j, filt in enumerate(filters):
        num = trapz(basis * filt[:, None], spectrum=None, axis=1)
        den = trapz(filt, spectrum=None)
        A[j, :] = num / den
    return A.T  # shape (N_b, N_f)

def reconstruct_spectrum(basis, filters, photometry, alpha=1.0):
    A = build_design_matrix(basis, filters)
    reg = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
    reg.fit(A, photometry)
    coef = reg.coef_
    recon_spec = np.dot(coef, basis)
    return recon_spec, coef

# --------------------- Main ---------------------
def main():
    # Wavelength grid
    wavelengths = np.linspace(400, 700, 301)  # nm

    # Basis functions
    n_basis = 5
    basis = generate_basis_functions(wavelengths, n_basis, seed=42)

    # True coefficients
    rng = np.random.default_rng(123)
    true_coef = rng.uniform(-1, 1, size=n_basis)

    # Synthetic spectrum
    true_spectrum = synthesize_spectrum(basis, true_coef, noise_std=0.01, seed=99)

    # Filters
    filter_centers = [450, 550, 650]   # nm
    filter_width = 30                  # nm
    filters = generate_filters(wavelengths, filter_centers, filter_width)

    # Photometric measurements
    photometry = compute_photometry(true_spectrum, filters)

    # Reconstruct spectrum
    recon_spectrum, recon_coef = reconstruct_spectrum(basis, filters, photometry, alpha=0.5)

    # Print results
    print("True coefficients:\n", true_coef)
    print("\nReconstructed coefficients:\n", recon_coef)
    print("\nPhotometric data (observed):\n", photometry)
    print("\nFirst 10 points of true spectrum vs reconstructed spectrum:")
    for t, r in zip(true_spectrum[:10], recon_spectrum[:10]):
        print(f"{t:.4f}  {r:.4f}")

if __name__ == "__main__":
    main()