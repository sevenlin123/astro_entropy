import numpy as np
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, coeffs):
    """Generate a spectrum from a linear combination of basis functions."""
    # Simple basis: Gaussian functions centered at fixed positions
    centers = np.array([4000, 5000, 6000])  # in Angstrom
    widths = np.array([300, 300, 300])
    spec = np.zeros_like(wavelengths, dtype=float)
    for c, w, a in zip(centers, widths, coeffs):
        spec += a * np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
    return spec

def generate_synthetic_spectra(n_samples, wavelengths):
    """Create synthetic spectra with random coefficients."""
    rng = np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, 3))
    spectra = np.array([spectral_model(wavelengths, coeffs[i]) for i in range(n_samples)])
    return spectra, coeffs

def photometry_from_spectrum(spectra, filters, wavelengths):
    """Simulate photometric measurements from spectra."""
    # Assume simple boxcar filter transmission
    fluxes = np.zeros((spectra.shape[0], len(filters)))
    for i, filt in enumerate(filters):
        mask = (wavelengths >= filt[0]) & (wavelengths <= filt[1])
        fluxes[:, i] = spectra[:, mask].mean(axis=1)
    return fluxes

def inverse_reconstruction(fluxes, wavelengths, filters, n_components=3):
    """
    Reconstruct spectrum using ridge regression (linear model).
      - fluxes: (n_samples, n_filters)
      - *.. ....
    """
    # 1: transform fluxes to pseudo-spectral points using filter response
    try:
        # Compute pseudo-spectra (i.e., using mean wavelengths for each filter
        #   + add......?????? .......