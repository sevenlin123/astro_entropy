#!/usr/bin/env python3
"""
Minimal spectral reconstruction from photometry.
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model ---------------------------------------------------------
# ----------------------------------------------------------------------
def wavelength_grid(start=4000, stop=7000, num=1000):
    """Return a linear wavelength array (Angstrom)."""
    return np.linspace(start, stop, num)


def gaussian_line(center, width, depth):
    """Return a Gaussian absorption line profile."""
    return 1 - depth * norm.pdf(x=wavelength_grid(), loc=center, scale=width)


def spectral_model(params):
    """
    Simple spectral model: flat continuum + two Gaussian absorption lines.
    Parameters are supplied as a dict with keys:
        'center1', 'width1', 'depth1',
        'center2', 'width2', 'depth2'
    """
    wav = wavelength_grid()
    continuum = np.ones_like(wav)

    line1 = gaussian_line(
        center=params["center1"],
        width=params["width1"],
        depth=params["depth1"],
    )
    line2 = gaussian_line(
        center=params["center2"],
        width=params["width2"],
        depth=params["depth2"],
    )

    spectrum = continuum * line1 * line2
    return spectrum


# ----------------------------------------------------------------------
# Synthetic data generation ---------------------------------------------
# ----------------------------------------------------------------------
def generate_filters(n_filters=5, center_start=4200, center_step=500):
    """Generate simple Gaussian filter transmission curves."""
    wav = wavelength_grid()
    filters = []
    for i in range(n_filters):
        center = center_start + i * center_step
        width = 200.0
        filt = norm.pdf(x=wav, loc=center, scale=width)
        filt /= filt.max()  # normalize
        filters.append(filt)
    return np.array(filters)


def compute_photometry(spectrum, filters):
    """
    Compute mean flux in each filter: integral(spectrum*filter)/integral(filter).
    Returns an array of photometric fluxes.
    """
    wav = wavelength_grid()
    phot = []
    for filt in filters:
        num = np.trapz(spectrum * filt, wav)
        denom = np.trapz(filt, wav)
        phot.append(num / denom)
    return np.array(phot)


def build_basis(n_basis=10):
    """
    Build a set of basis spectra by sampling parameters randomly.
    Returns list of spectra and corresponding parameter dicts.
    """
    rng = np.random.default_rng(seed=42)
    basis = []
    params_list = []
    for _ in range(n_basis):
        params = {
            "center1": rng.uniform(4800, 5200),
            "width1": rng.uniform(10, 40),
            "depth1": rng.uniform(0.1, 0.4),
            "center2": rng.uniform(5800, 6200),
            "width2": rng.uniform(10, 40),
            "depth2": rng.uniform(0.1, 0.4),
        }
        spec = spectral_model(params)
        basis.append(spec)
        params_list.append(params)
    return basis, params_list


# ----------------------------------------------------------------------
# Reconstruction --------------------------------------------------------
# ----------------------------------------------------------------------
def build_basis_photometry(basis, filters):
    """Compute photometry of each basis spectrum."""
    phot_list = []
    for spec in basis:
        phot_list.append(compute_photometry(spec, filters))
    return np.array(phot_list).T  # shape (n_filters, n_basis)


def reconstruct_spectrum(photometry, basis_phot, basis):
    """
    Reconstruct the spectrum by solving a linear least‑squares problem.
    Uses ridge regression to regularise.
    """
    # Ridge regression
    reg = Ridge(alpha=1e-3, fit_intercept=False, solver="auto")
    reg.fit(basis_phot.T, photometry)
    coeffs = reg.coef_
    # Combine basis spectra
    recon = np.zeros_like(basis[0])
    for c, spec in zip(coeffs, basis):
        recon += c * spec
    return recon, coeffs


# ----------------------------------------------------------------------
# Main ------------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    np.random.seed(0)

    # True spectrum
    true_params = {
        "center1": 5050,
        "width1": 30,
        "depth1": 0.25,
        "center2": 6100,
        "width2": 35,
        "depth2": 0.3,
    }
    true_spec = spectral_model(true_params)

    # Filters
    filters = generate_filters(n_filters=6)

    # Photometry of true spectrum
    phot_true = compute_photometry(true_spec, filters)

    # Basis set
    basis, _ = build_basis(n_basis=12)
    basis_phot = build_basis_photometry(basis, filters)

    # Reconstruction
    recon_spec, coeffs = reconstruct_spectrum(phot_true, basis_phot, basis)

    # Evaluate
    rmse = np.sqrt(np.mean((recon_spec - true_spec) ** 2))
    print(f"RMSE between true and reconstructed spectrum: {rmse:.6f}")

    # Print first few coefficients
    print("First 5 reconstruction coefficients:")
    print(coeffs[:5])


if __name__ == "__main__":
    main()