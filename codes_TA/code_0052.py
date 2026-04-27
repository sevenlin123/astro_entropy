#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression


def wavelength_grid(start=3500.0, stop=7500.0, num=1000):
    """Return an evenly spaced wavelength grid."""
    return np.linspace(start, stop, num)


def gaussian_template(wave, amp, mu, sigma):
    """Return a Gaussian-shaped template."""
    return amp * np.exp(-0.5 * ((wave - mu) / sigma) ** 2)


def generate_templates(n_templates, wave, rng=None):
    """
    Generate `n_templates` synthetic spectral templates on the given wavelength grid.
    Each template is a Gaussian with random amplitude, center, and width.
    """
    rng = rng or np.random.default_rng()
    templates = []
    for _ in range(n_templates):
        amp = rng.uniform(0.5, 1.5)
        mu = rng.uniform(wave.min() + 0.1 * (wave.max() - wave.min()),
                         wave.max() - 0.1 * (wave.max() - wave.min()))
        sigma = rng.uniform(50.0, 300.0)
        templates.append(gaussian_template(wave, amp, mu, sigma))
    return np.array(templates)  # shape (n_templates, n_wave)


def generate_filters(n_filters, wave, rng=None):
    """
    Generate `n_filters` synthetic filter transmission curves on the wavelength grid.
    Each filter is a Gaussian transmission curve.
    """
    rng = rng or np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        mu = rng.uniform(wave.min() + 0.1 * (wave.max() - wave.min()),
                         wave.max() - 0.1 * (wave.max() - wave.min()))
        sigma = rng.uniform(150.0, 500.0)
        filt = np.exp(-0.5 * ((wave - mu) / sigma) ** 2)
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, n_wave)


def synthetic_spectrum(templates, coeffs=None, noise_sigma=0.0, rng=None):
    """
    Construct a synthetic spectrum as a linear combination of templates.
    If `coeffs` is None, random positive weights are used.
    Optional Gaussian noise is added.
    """
    rng = rng or np.random.default_rng()
    if coeffs is None:
        coeffs = rng.uniform(0.5, 1.5, size=templates.shape[0])
    spec = coeffs @ templates  # shape (n_wave,)
    if noise_sigma > 0.0:
        spec += rng.normal(0.0, noise_sigma, size=spec.size)
    return spec, coeffs


def photometry_from_spectrum(spectrum, filters):
    """
    Compute synthetic photometric fluxes by integrating spectrum through each filter.
    """
    # Simple discrete convolution: sum(spectrum * filter) / sum(filter)
    fluxes = (filters * spectrum).sum(axis=1) / filters.sum(axis=1)
    return fluxes  # shape (n_filters,)


def reconstruct_spectrum_from_photometry(filters, photometry, templates):
    """
    Recover the spectrum by fitting a linear combination of templates to the
    observed photometric fluxes. Returns the estimated spectrum and coefficients.
    """
    # Build design matrix: each row corresponds to a filter, each column to a template
    design_matrix = (filters[:, :, None] * templates[None, :, :]).sum(axis=1)
    # Fit linear regression without intercept (pure linear combination)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design_matrix, photometry)
    coeffs_est = lr.coef_
    spec_est = coeffs_est @ templates
    return spec_est, coeffs_est


def main():
    rng = np.random.default_rng(seed=42)

    # 1. Wavelength grid
    wave = wavelength_grid()

    # 2. Spectral model: generate templates
    n_templ = 5
    templates = generate_templates(n_templ, wave, rng=rng)

    # 3. Generate synthetic spectrum
    true_spec, true_coeffs = synthetic_spectrum(templates, rng=rng, noise_sigma=0.0)

    # 4. Generate photometric data
    n_filters = 4
    filters = generate_filters(n_filters, wave, rng=rng)
    phot = photometry_from_spectrum(true_spec, filters)

    # Add measurement noise to photometry
    phot_noisy = phot + rng.normal(0.0, 0.02, size=phot.size)

    # 5. Reconstruction
    recon_spec, recon_coeffs = reconstruct_spectrum_from_photometry(
        filters, phot_noisy, templates
    )

    # 6. Output results
    print("True coefficients:\n", true_coeffs)
    print("\nReconstructed coefficients:\n", recon_coeffs)
    print("\nSpectral reconstruction residual (L2 norm):",
          np.linalg.norm(true_spec - recon_spec))

    # Optional: compare spectra visually if desired (commented out)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(wave, true_spec, label="True Spectrum")
    # plt.plot(wave, recon_spec, '--', label="Reconstructed Spectrum")
    # plt.xlabel("Wavelength (Å)")
    # plt.ylabel("Flux (arbitrary units)")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()