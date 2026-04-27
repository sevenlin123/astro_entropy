#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.

1. Build a simple spectral model.
2. Generate synthetic spectra.
3. Create synthetic photometry from those spectra.
4. Train a linear regressor to map photometry → spectrum.
5. Reconstruct a test spectrum from its photometry.
"""

import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# 1. Define the spectral grid and filter set
# --------------------------------------------------------------------
def wavelength_grid(n_lam=1000, lam_min=300.0, lam_max=800.0):
    """Return a linearly spaced wavelength array (nm)."""
    return np.linspace(lam_min, lam_max, n_lam)

def gaussian_filter(lam, center, width, amplitude=1.0):
    """Simple Gaussian filter response."""
    return amplitude * np.exp(-0.5 * ((lam - center) / width)**2)

def filter_set(lam):
    """Return a list of filter responses evaluated on `lam`."""
    filters = {
        "U": gaussian_filter(lam, center=350, width=30),
        "B": gaussian_filter(lam, center=440, width=40),
        "V": gaussian_filter(lam, center=550, width=40),
        "R": gaussian_filter(lam, center=640, width=50),
        "I": gaussian_filter(lam, center=790, width=60),
    }
    return [filters[key] for key in ["U","B","V","R","I"]]

# --------------------------------------------------------------------
# 2. Construct synthetic spectral templates
# --------------------------------------------------------------------
def random_spectrum(lam, n_peaks=3, seed=None):
    """
    Generate a random stellar spectrum as a sum of Gaussians.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.zeros_like(lam)
    for _ in range(n_peaks):
        amp   = rng.uniform(0.5, 1.5)
        cen   = rng.uniform(lam[0], lam[-1])
        wid   = rng.uniform(10.0, 30.0)
        spectrum += gaussian_filter(lam, cen, wid, amplitude=amp)
    # Add weak continuum
    spectrum += 0.2 * rng.normal(size=lam.size)
    return spectrum

def generate_synthetic_library(n_models=200, lam=None, seed=42):
    """
    Produce a library of synthetic spectra.
    """
    rng = np.random.default_rng(seed)
    spectra = []
    for i in range(n_models):
        spectra.append(random_spectrum(lam, n_peaks=3, seed=rng.integers(1e6)))
    return np.array(spectra)

# --------------------------------------------------------------------
# 3. Compute synthetic photometry
# --------------------------------------------------------------------
def compute_photometry(spectra, filters, lam):
    """
    Integrate each spectrum over each filter to get a photometric vector.
    """
    phot = np.empty((spectra.shape[0], len(filters)))
    for j, filt in enumerate(filters):
        # Numerical integration over wavelength
        flux = simps(spectra * filt, lam, axis=1)
        phot[:, j] = flux
    return phot

# --------------------------------------------------------------------
# 4. Train regressor (Ridge) to map photometry → spectrum
# --------------------------------------------------------------------
def train_reconstructor(X_train, Y_train, alpha=1.0):
    """
    Fit a Ridge regression model that predicts full spectrum from photometry.
    """
    model = Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X_train, Y_train)
    return model

# --------------------------------------------------------------------
# 5. Reconstruction routine
# --------------------------------------------------------------------
def reconstruct_spectrum(model, photometry):
    """
    Predict a full spectrum from its photometric measurements.
    """
    return model.predict(photometry.reshape(1, -1))[0]

# --------------------------------------------------------------------
# 6. Main demo
# --------------------------------------------------------------------
def main():
    # Set up wavelength and filters
    lam = wavelength_grid()
    filters = filter_set(lam)

    # Generate synthetic spectral library
    spectra_lib = generate_synthetic_library(n_models=200, lam=lam)

    # Compute photometry for the library
    photometry_lib = compute_photometry(spectra_lib, filters, lam)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        photometry_lib, spectra_lib, test_size=0.2, random_state=0
    )

    # Train reconstructor
    reconstructor = train_reconnector(X_train, y_train, alpha=1.0)

    # Pick a test spectrum and reconstruct it
    idx = 0  # first test sample
    true_spec = y_test[idx]
    phot = X_test[idx]
    reconstructed_spec = reconstruct_spectrum(reconstructor, phot)

    # Simple evaluation
    mse = np.mean((true_spec - reconstructed_spec)**2)
    print(f"Reconstruction MSE: {mse:.4f}")

    # Optional: compare a few points
    print("\nWavelength (nm)\tTrue\tReconstructed")
    for w, t, r in zip(lam[::100], true_spec[::100], reconstructed_spec[::100]):
        print(f"{w:8.1f}\t{t:8.4f}\t{r:12.4f}")

if __name__ == "__main__":
    main()