#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def spectral_model():
    """
    Return wavelengths and a set of basis spectra (e.g., black‑body, line, continuum).
    """
    lam = np.linspace(3000, 10000, 500)  # Å
    # Basis 1: flat continuum
    b1 = np.ones_like(lam)
    # Basis 2: Gaussian emission line at 5000 Å
    b2 = np.exp(-0.5 * ((lam - 5000) / 200)**2)
    # Basis 3: A black‑body curve at T=6000 K
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam_m = lam * 1e-10
    B = (2*h*c**2) / (lam_m**5 * (np.exp(h*c/(lam_m*k*6000)) - 1))
    b3 = B / np.max(B)
    basis = np.vstack([b1, b2, b3]).T  # shape (n_lam, n_basis)
    return lam, basis


# ---------- Bandpasses ----------
def generate_bandpasses(lam):
    """
    Create a few simple bandpasses (top‑hat) over the wavelength grid.
    Returns a list of (band_name, response_array).
    """
    bands = []
    for center, width in [(3500, 400), (4500, 500), (5500, 600), (6500, 400), (7500, 500)]:
        resp = np.zeros_like(lam)
        mask = (lam >= center - width/2) & (lam <= center + width/2)
        resp[mask] = 1.0
        bands.append((f"Band_{center}", resp))
    return bands


# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(n_samples, basis, noise_std=0.01, random_state=42):
    rng = np.random.default_rng(random_state)
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return coeffs, spectra


# ---------- Photometry ----------
def synthesize_photometry(spectra, bands, lam):
    """
    Compute band fluxes by integrating spectrum × response.
    """
    phot = np.empty((spectra.shape[0], len(bands)))
    for i, (_, resp) in enumerate(bands):
        integrand = spectra * resp  # broadcast over samples
        band_flux = simps(integrand, lam, axis=1) / simps(resp, lam)
        phot[:, i] = band_flux
    return phot


# ---------- Reconstruction ----------
def reconstruct_spectra(phot, bands, lam, basis):
    """
    Reconstruct spectra from photometry using linear least squares on basis functions.
    """
    # Build design matrix: integral of each basis × band response
    X = np.empty((len(bands), basis.shape[1]))
    for i, (_, resp) in enumerate(bands):
        integrand = basis * resp  # shape (n_lam, n_basis)
        X[i, :] = simps(integrand, lam, axis=0) / simps(resp, lam)

    # Fit coefficients
    lr = LinearRegression(fit_intercept=False).fit(X.T, phot.T)
    coeffs_rec = lr.coef_.T  # shape (n_samples, n_basis)

    # Reconstruct spectra
    spectra_rec = coeffs_rec @ basis.T
    return spectra_rec


# ---------- Demo ----------
if __name__ == "__main__":
    lam, basis = spectral_model()
    bands = generate_bandpasses(lam)
    true_coeffs, spectra_true = generate_synthetic_spectra(10, basis)
    phot = synthesize_photometry(spectra_true, bands, lam)
    spectra_rec = reconstruct_spectra(phot, bands, lam, basis)

    # Print a small comparison for the first spectrum
    idx = 0
    print("True coefficients:", true_coeffs[idx])
    rec_coeffs = (spectra_rec[idx] @ basis).sum() / (basis.T @ basis).diagonal()
    print("Reconstructed coefficients (approx):", rec_coeffs)
    print("Difference (true - rec):", true_coeffs[idx] - rec_coeffs)