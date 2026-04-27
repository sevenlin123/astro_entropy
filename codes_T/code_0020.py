#!/usr/bin/env python3
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

# -----------------------------
# Spectral model utilities
# -----------------------------
def gaussian(x, mu, sigma):
    """One-dimensional Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def basis_spectra(wave, n_basis=5, seed=None):
    """
    Generate a collection of Gaussian basis spectra.
    Returns a matrix of shape (n_basis, len(wave)).
    """
    rng = np.random.default_rng(seed)
    mus = rng.uniform(wave.min(), wave.max(), size=n_basis)
    sigmas = rng.uniform(20, 100, size=n_basis)  # nm
    bases = np.array([gaussian(wave, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    return bases

def synth_spectrum(bases, coeffs=None):
    """
    Construct a synthetic spectrum as a linear combination of basis functions.
    If coeffs is None, draw random coefficients.
    """
    if coeffs is None:
        coeffs = np.random.uniform(-1, 1, size=bases.shape[0])
    spectrum = coeffs @ bases
    return spectrum

# -----------------------------
# Filter generation
# -----------------------------
def top_hat_filter(wave, center, width):
    """Simple top-hat filter."""
    return ((wave >= center - width/2) & (wave <= center + width/2)).astype(float)

def generate_filters(wave, centers=[350, 450, 550, 650], width=80):
    """Create a list of filter transmissions."""
    return [top_hat_filter(wave, c, width) for c in centers]

# -----------------------------
# Photometry simulation
# -----------------------------
def photometry_from_spectrum(spectrum, filters):
    """Compute synthetic photometric fluxes for a single spectrum."""
    return np.array([np.trapz(spectrum * filt, dx=wave[1]-wave[0]) for filt in filters])

def photometry_dataset(n_samples, wave, filters):
    """Generate a dataset of spectra and corresponding photometries."""
    bases = basis_spectra(wave, seed=42)
    spectra = []
    photometries = []
    rng = np.random.default_rng(123)
    for _ in range(n_samples):
        coeffs = rng.uniform(-1, 1, size=bases.shape[0])
        spec = synth_spectrum(bases, coeffs)
        spectra.append(spec)
        photometries.append(photometry_from_spectrum(spec, filters))
    return np.array(spectra), np.array(photometries)

# -----------------------------
# Reconstruction framework
# -----------------------------
def compute_pseudoinverse(photometries, spectra):
    """
    Compute the pseudo-inverse of the mapping from spectra to photometry.
    photometries : (N, P)
    spectra     : (N, S)
    Returns an (S, P) matrix that maps photometry to spectra.
    """
    M = photometries.T @ spectra
    pinv_M = pinv(M)
    return pinv_M

def reconstruct_spectrum(photometry, pinv_M):
    """Estimate spectrum from photometry using the precomputed pseudo-inverse."""
    return pinv_M @ photometry

# -----------------------------
# Main routine
# -----------------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(300, 2500, 1200)  # nm

    # Filters
    filters = generate_filters(wave)

    # Training set
    N_train = 400
    spectra_train, phot_train = photometry_dataset(N_train, wave, filters)

    # Compute pseudo-inverse
    pinv_M = compute_pseudoinverse(phot_train, spectra_train)

    # Test set
    N_test = 5
    spectra_true, phot_test = photometry_dataset(N_test, wave, filters)

    # Reconstruction
    spectra_rec = np.array([reconstruct_spectrum(p, pinv_M) for p in phot_test])

    # Plot results for one test case
    idx = 0
    plt.figure(figsize=(8, 4))
    plt.plot(wave, spectra_true[idx], label="True spectrum")
    plt.plot(wave, spectra_rec[idx], '--', label="Reconstructed spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.title("Spectral Reconstruction Example")
    plt.legend()
    plt.tight_layout()
    plt.show()