#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Define the wavelength grid
# ----------------------------------------------------------------------
def get_wavelength_grid():
    """Return an evenly spaced wavelength grid in Å."""
    return np.linspace(3000, 25000, 1000)

# ----------------------------------------------------------------------
# 2. Define filter responses
# ----------------------------------------------------------------------
def get_filter_responses(wavelengths):
    """
    Return a dictionary of simple top‑hat filter responses.
    Each response is a 1‑D array with the same shape as `wavelengths`.
    """
    filters = {
        'u':  np.logical_and(wavelengths >= 3500, wavelengths <= 4000).astype(float),
        'g':  np.logical_and(wavelengths >= 4000, wavelengths <= 5500).astype(float),
        'r':  np.logical_and(wavelengths >= 5500, wavelengths <= 7000).astype(float),
        'i':  np.logical_and(wavelengths >= 7000, wavelengths <= 8500).astype(float),
        'z':  np.logical_and(wavelengths >= 8500, wavelengths <= 10000).astype(float)
    }
    return filters

# ----------------------------------------------------------------------
# 3. Generate synthetic spectra
# ----------------------------------------------------------------------
def gauss(x, amp, cen, wid):
    """One Gaussian component."""
    return amp * np.exp(-0.5 * ((x - cen) / wid)**2)

def generate_synthetic_spectra(n_spectra, wavelengths, rng=None):
    """
    Generate synthetic spectra as sums of random Gaussian components.
    
    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    wavelengths : ndarray
        Wavelength grid (Å).
    rng : np.random.Generator, optional
        Random number generator instance.
        
    Returns
    -------
    spectra : ndarray
        Array of shape (n_spectra, len(wavelengths)).
    """
    if rng is None:
        rng = np.random.default_rng()
    spectra = np.zeros((n_spectra, len(wavelengths)))
    for i in range(n_spectra):
        n_comp = rng.integers(2, 5)
        amps = rng.uniform(0.5, 2.0, size=n_comp)
        cents = rng.uniform(4000, 20000, size=n_comp)
        wids  = rng.uniform(200, 2000, size=n_comp)
        spec = np.sum([gauss(wavelengths, a, c, w) for a, c, w in zip(amps, cents, wids)], axis=0)
        # Normalize to unit flux
        spec /= trapz(spec, wavelengths)
        spectra[i] = spec
    return spectra

# ----------------------------------------------------------------------
# 4. Compute photometric fluxes
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate each spectrum through each filter response.
    
    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, n_wave)
    filters : dict of {name: response}
    wavelengths : ndarray
    
    Returns
    -------
    photometry : ndarray, shape (n_spectra, n_filters)
    """
    n_spectra = spectra.shape[0]
    n_filters = len(filters)
    photometry = np.empty((n_spectra, n_filters))
    filter_names = list(filters.keys())
    for j, name in enumerate(filter_names):
        resp = filters[name]
        # Integrated flux: ∫ S(λ) R(λ) dλ / ∫ R(λ) dλ
        integ = trapz(spectra * resp, wavelengths, axis=1)
        norm = trapz(resp, wavelengths)
        photometry[:, j] = integ / norm
    return photometry, filter_names

# ----------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum(photometry, wavelengths, alpha=1.0):
    """
    Train a linear model to map photometry to full spectrum.
    
    Parameters
    ----------
    photometry : ndarray, shape (n_samples, n_filters)
    wavelengths : ndarray
    alpha : float
        Regularization strength for Ridge regression.
    
    Returns
    -------
    model : Ridge object
        Trained model.
    """
    n_samples = photometry.shape[0]
    # For training we need the target spectra
    # In this minimal example we will use a global dataset
    # The user must supply the corresponding spectra array
    # Here we demonstrate by generating a synthetic dataset on the fly.
    rng = np.random.default_rng(seed=42)
    spectra = generate_synthetic_spectra(n_samples, wavelengths, rng=rng)
    model = Ridge(alpha=alpha)
    model.fit(photometry, spectra)
    return model

# ----------------------------------------------------------------------
# 6. Demo pipeline
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=12345)
    wavelengths = get_wavelength_grid()
    filters = get_filter_responses(wavelengths)
    
    # Generate a synthetic spectrum to reconstruct
    true_spectrum = generate_synthetic_spectra(1, wavelengths, rng=rng)[0]
    
    # Compute its photometry
    photometry, filter_names = compute_photometry(
        true_spectrum.reshape(1, -1), filters, wavelengths
    )
    
    # Build a training set (other synthetic spectra + their photometry)
    n_train = 50
    train_spectra = generate_synthetic_spectra(n_train, wavelengths, rng=rng)
    train_photometry, _ = compute_photometry(train_spectra, filters, wavelengths)
    
    # Train reconstruction model
    model = Ridge(alpha=1.0)
    model.fit(train_photometry, train_spectra)
    
    # Predict spectrum from photometry
    reconstructed = model.predict(photometry)[0]
    
    # Evaluate reconstruction error (RMSE)
    rmse = np.sqrt(np.mean((reconstructed - true_spectrum)**2))
    print(f"Reconstruction RMSE: {rmse:.4f}")
    
    # Optional: plot (commented out per instruction)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,4))
    # plt.plot(wavelengths, true_spectrum, label='True')
    # plt.plot(wavelengths, reconstructed, '--', label='Reconstructed')
    # plt.xlabel('Wavelength (Å)')
    # plt.ylabel('Normalized Flux')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()