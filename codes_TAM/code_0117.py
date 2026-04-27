#!/usr/bin/env python3
"""
Minimal reconstruction framework:
- Define a spectral basis
- Generate synthetic spectra
- Produce photometric measurements
- Reconstruct spectra from photometry
"""

import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def wavelength_grid(start=400.0, stop=700.0, n=200):
    """Return a linear wavelength grid in nm."""
    return np.linspace(start, stop, n)


def basis_functions(n_basis, wavelengths):
    """
    Simple polynomial basis up to order n_basis-1.
    Returns an array of shape (n_basis, len(wavelengths)).
    """
    X = np.vstack([wavelengths**i for i in range(n_basis)])
    # Normalise each basis vector
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def gaussian_filter(center, width, wavelengths):
    """Return a Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def filter_set(n_filters, wavelengths):
    """
    Build a set of simple Gaussian filters spaced across the grid.
    Returns a list of filter transmissions.
    """
    centers = np.linspace(wavelengths.min() + 0.2 * wavelengths.ptp(),
                          wavelengths.max() - 0.2 * wavelengths.ptp(),
                          n_filters)
    width = 30.0  # nm
    return [gaussian_filter(c, width, wavelengths) for c in centers]


def synthetic_spectra(n_samples, basis, coeff_range=(0.5, 1.5), rng=None):
    """Generate random spectra as linear combinations of basis."""
    rng = rng or np.random.default_rng()
    n_basis, n_wave = basis.shape
    coeffs = rng.uniform(coeff_range[0], coeff_range[1],
                         size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs


def photometric_fluxes(spectra, filters):
    """Integrate spectra over each filter to get fluxes."""
    n_samples, n_wave = spectra.shape
    n_filters = len(filters)
    flux = np.empty((n_samples, n_filters))
    for i, filt in enumerate(filters):
        flux[:, i] = simps(spectra * filt, axis=1)
    return flux


def magnitude_from_flux(flux, zeropoint=25.0):
    """Convert flux to AB magnitudes."""
    return -2.5 * np.log10(flux) + zeropoint


def flux_from_magnitude(mag, zeropoint=25.0):
    """Inverse operation."""
    return 10 ** ((zeropoint - mag) / 2.5)


def train_regressor(X, Y, alpha=1.0):
    """Train a ridge regressor to map photometry to spectral coeffs."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, Y)
    return reg


def reconstruct_spectrum(regressor, photometry, basis):
    """Predict spectrum from photometric data."""
    coeffs_pred = regressor.predict(photometry)
    spectrum_pred = coeffs_pred @ basis
    return spectrum_pred, coeffs_pred


def main():
    rng = np.random.default_rng(42)

    # 1. Define spectral model
    wavelengths = wavelength_grid()
    n_basis = 5
    basis = basis_functions(n_basis, wavelengths)

    # 2. Generate synthetic spectra
    n_train = 500
    n_test = 50
    spectra_all, coeffs_all = synthetic_spectra(
        n_train + n_test, basis, rng=rng)

    spectra_train, spectra_test = np.split(spectra_all,
                                           [n_train])
    coeffs_train, coeffs_test = np.split(coeffs_all,
                                         [n_train])

    # 3. Photometric data
    filters = filter_set(4, wavelengths)
    flux_train = photometric_fluxes(spectra_train, filters)
    flux_test = photometric_fluxes(spectra_test, filters)

    # Convert to magnitudes (optional)
    mags_train = magnitude_from_flux(flux_train)
    mags_test = magnitude_from_flux(flux_test)

    # 4. Reconstruction
    reg = train_regressor(mags_train, coeffs_train, alpha=1.0)

    spectra_recon, coeffs_recon = reconstruct_spectrum(reg, mags_test, basis)

    # Evaluation
    err = np.mean((spectra_test - spectra_recon) ** 2, axis=1)
    print(f"Mean squared error per test spectrum: {err.mean():.6f}")

    # Example plot (requires matplotlib, optional)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, spectra_test[idx], label="True")
        plt.plot(wavelengths, spectra_recon[idx], '--', label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()