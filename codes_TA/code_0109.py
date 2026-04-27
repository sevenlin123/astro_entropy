#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# Spectral model – a set of Gaussian basis functions
# ------------------------------------------------------------------
def basis_functions(wl, n_basis=10):
    """Return basis matrix of shape (len(wl), n_basis)."""
    wl = np.asarray(wl)
    centers = np.linspace(wl.min() + 0.1*wl.size,
                          wl.max() - 0.1*wl.size,
                          n_basis)
    sigma = 20.0  # nm
    B = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / sigma) ** 2)
    # Normalize each basis function to unit area
    B /= simps(B, wl, axis=0)[None, :]
    return B


# ------------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wl, n_basis=10, noise_level=0.02, seed=None):
    """Return spectra (n_samples, len(wl)) and true coefficients."""
    rng = np.random.default_rng(seed)
    B = basis_functions(wl, n_basis)
    coeffs = rng.standard_normal(size=(n_samples, n_basis))
    spectra = coeffs @ B.T
    # Add small Gaussian noise
    spectra += noise_level * rng.standard_normal(size=spectra.shape)
    return spectra, coeffs


# ------------------------------------------------------------------
# Define photometric bandpasses (Gaussian filters)
# ------------------------------------------------------------------
def bandpasses(band_defs, wl):
    """Return list of band response arrays matching wl."""
    responses = []
    for name, center, width in band_defs:
        resp = np.exp(-0.5 * ((wl - center) / width) ** 2)
        resp /= simps(resp, wl)  # normalize to unit integral
        responses.append((name, resp))
    return responses


# ------------------------------------------------------------------
# Generate photometric measurements
# ------------------------------------------------------------------
def photometric_fluxes(spectra, band_responses, wl):
    """Return photometry array (n_samples, n_bands)."""
    n_samples = spectra.shape[0]
    n_bands = len(band_responses)
    fluxes = np.empty((n_samples, n_bands))
    for i, (name, resp) in enumerate(band_responses):
        # Integrate flux * response over wavelength
        fluxes[:, i] = simps(spectra * resp, wl, axis=1)
    return fluxes


# ------------------------------------------------------------------
# Reconstruct spectra from photometry
# ------------------------------------------------------------------
def reconstruct_spectra(photometry, band_responses, wl, n_basis=10):
    """Return reconstructed spectra (n_samples, len(wl))."""
    # Precompute integrated basis matrix
    B = basis_functions(wl, n_basis)                # (len(wl), n_basis)
    M = np.empty((len(band_responses), n_basis))     # (n_bands, n_basis)
    for i, (_, resp) in enumerate(band_responses):
        M[i, :] = simps(B * resp[:, None], wl, axis=0)

    n_samples = photometry.shape[0]
    coeffs_hat = np.empty((n_samples, n_basis))
    for s in range(n_samples):
        y = photometry[s, :]
        # Solve least‑squares M @ c = y
        coeffs_hat[s, :] = np.linalg.lstsq(M, y, rcond=None)[0]

    recon_spectra = coeffs_hat @ B.T
    return recon_spectra, coeffs_hat


# ------------------------------------------------------------------
# Main routine – generate data, reconstruct, evaluate
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400.0, 800.0, 1000)  # nm

    # Generate synthetic spectra
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, wl, n_basis=10, noise_level=0.01, seed=42)

    # Define photometric bands: (name, center, width)
    band_defs = [
        ("U", 360.0, 20.0),
        ("B", 440.0, 25.0),
        ("V", 550.0, 30.0),
        ("R", 640.0, 30.0),
        ("I", 790.0, 30.0)
    ]
    bands = bandpasses(band_defs, wl)

    # Generate photometry
    photometry = photometric_fluxes(spectra_true, bands, wl)

    # Reconstruct spectra
    spectra_recon, coeffs_est = reconstruct_spectra(
        photometry, bands, wl, n_basis=10)

    # Evaluate reconstruction accuracy
    rmse = np.sqrt(mean_squared_error(spectra_true, spectra_recon))
    print(f"Reconstruction RMSE: {rmse:.4f}")