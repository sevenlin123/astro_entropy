#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# 1. Define spectral model (sum of Gaussians)
# --------------------------------------------------------------------------- #

def gaussian_spectrum(wave, amps, means, sigmas):
    """
    Construct a spectrum as a sum of Gaussian components.

    Parameters
    ----------
    wave   : array_like
        Wavelength grid.
    amps   : array_like
        Amplitudes of the Gaussian components.
    means  : array_like
        Centers of the Gaussian components.
    sigmas : array_like
        Standard deviations of the Gaussian components.

    Returns
    -------
    spec : ndarray
        Spectrum evaluated at `wave`.
    """
    spec = np.zeros_like(wave)
    for a, m, s in zip(amps, means, sigmas):
        spec += a * norm.pdf(wave, loc=m, scale=s)
    return spec


def generate_parameters(n_samples, n_components=3):
    """
    Randomly draw parameters for synthetic spectra.
    """
    rng = np.random.default_rng(seed=42)
    amps = rng.uniform(0.5, 1.5, size=(n_samples, n_components))
    means = rng.uniform(500, 700, size=(n_samples, n_components))
    sigmas = rng.uniform(20, 40, size=(n_samples, n_components))
    return amps, means, sigmas


def generate_synthetic_spectra(n_samples, wave, n_components=3, noise_std=0.02):
    """
    Generate noisy synthetic spectra.
    """
    amps, means, sigmas = generate_parameters(n_samples, n_components)
    spectra = np.array([gaussian_spectrum(wave, a, m, s)
                       for a, m, s in zip(amps, means, sigmas)])
    noise = noise_std * np.random.randn(*spectra.shape)
    return spectra + noise


# --------------------------------------------------------------------------- #
# 2. Define filters and generate photometry
# --------------------------------------------------------------------------- #

def create_filter_grid(wave, lower, upper):
    """
    Simple rectangular filter between `lower` and `upper` (nm).
    """
    filt = np.where((wave >= lower) & (wave <= upper), 1.0, 0.0)
    return filt


def make_filters(wave):
    """Return a dictionary of three rectangular filters."""
    return {
        'U': create_filter_grid(wave, 350, 450),
        'B': create_filter_grid(wave, 450, 550),
        'V': create_filter_grid(wave, 550, 650),
    }


def photometry_from_spectrum(spectrum, filters, wave):
    """
    Compute synthetic photometric measurements from a spectrum.
    """
    dw = wave[1] - wave[0]
    ph = {}
    for name, filt in filters.items():
        # weighted integral of spectrum over filter
        integrand = spectrum * filt
        val = np.sum(integrand) * dw / np.sum(filt) * dw
        ph[name] = val
    return np.array([ph[k] for k in sorted(ph)])


def generate_photometry(spectra, filters, wave):
    """Apply photometry to an array of spectra."""
    return np.array([photometry_from_spectrum(s, filters, wave) for s in spectra])


# --------------------------------------------------------------------------- #
# 3. Reconstruction from photometry
# --------------------------------------------------------------------------- #

def build_design_matrix(filters, wave):
    """
    Build the linear mapping from spectrum (at each wavelength) to photometric
    measurements. The design matrix has shape (n_filters, n_wave).
    """
    dw = wave[1] - wave[0]
    mat = []
    for filt in filters.values():
        col = filt * dw / np.sum(filt)  # normalization factor
        mat.append(col)
    return np.vstack(mat)  # shape (n_filters, n_wave)


def reconstruct_spectra_from_photometry(photometry, filters, wave, alpha=1.0):
    """
    Reconstruct spectra from photometric measurements using ridge regression.
    """
    X = build_design_matrix(filters, wave)          # (n_filt, n_wave)
    y = photometry                                # (n_samples, n_filt)

    # Transpose X to match sklearn convention: samples x features
    # Here each "sample" is the spectrum at a given wavelength.
    # We solve y = yhat = X @ coeff + noise, where coeff has shape (n_samples, n_wave)
    # Instead, we perform regression per wavelength: for each wavelength,
    # we fit coeff[*, i] to y using X[i,:].
    n_samples, n_filt = y.shape
    n_wave = X.shape[1]

    coeffs = np.zeros((n_samples, n_wave))
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    for i in range(n_wave):
        ridge.fit(X[i, :].reshape(-1, 1), y.T[i])
        coeffs[:, i] = ridge.coef_.ravel()
    return coeffs


# --------------------------------------------------------------------------- #
# 4. Main routine and evaluation
# --------------------------------------------------------------------------- #

def main():
    # Define wavelength grid
    wave = np.linspace(300, 800, 501)  # nm

    # Generate synthetic spectra
    n_samples = 50
    spectra = generate_synthetic_spectra(n_samples, wave, noise_std=0.01)

    # Define filters
    filters = make_filters(wave)

    # Compute photometric measurements
    photometry = generate_photometry(spectra, filters, wave)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra_from_photometry(photometry, filters, wave, alpha=0.1)

    # Evaluate reconstruction error
    rmse = np.sqrt(np.mean((spectra - recon_spectra)**2))
    print(f"Reconstruction RMSE: {rmse:.4f}")

    # Optionally display first spectrum comparison
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(10, 4))
    plt.plot(wave, spectra[idx], label="True")
    plt.plot(wave, recon_spectra[idx], '--', label="Reconstructed")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.title("Spectrum Reconstruction Example")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()