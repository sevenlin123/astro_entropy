#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import qr

def wavelength_grid(start, end, n):
    """Return evenly spaced wavelength array."""
    return np.linspace(start, end, n)

def orthonormal_basis(n_basis, n_wav, random_state=None):
    """Generate an orthonormal basis of size (n_basis, n_wav)."""
    rng = np.random.default_rng(random_state)
    mat = rng.standard_normal((n_basis, n_wav))
    q, _ = qr(mat, mode='economic')
    return q

def gaussian_filter(wav, center, width):
    """Return a single Gaussian filter response over wavelengths."""
    return np.exp(-0.5 * ((wav - center) / width) ** 2)

def generate_filters(n_filters, wav, width, random_state=None):
    """Generate a set of Gaussian filters."""
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(wav[0], wav[-1], n_filters)
    filters = np.array([gaussian_filter(wav, c, width) for c in centers])
    return filters

def generate_spectra(n_specs, basis, coeff_bounds=(0.0, 1.0),
                     noise_std=0.0, random_state=None):
    """Generate synthetic spectra as random linear combinations of the basis."""
    rng = np.random.default_rng(random_state)
    coeffs = rng.uniform(coeff_bounds[0], coeff_bounds[1], (n_specs, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_specs, n_wav)
    if noise_std > 0:
        spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

def photometry_from_spectra(spectra, filters):
    """Compute synthetic photometric measurements."""
    # specta (nspec, nwav), filters (nfilters, nwav)
    return spectra @ filters.T  # shape (nspec, nfilters)

def reconstruct_coefficients(phot, filters, basis, alpha=1.0):
    """Recover coefficients via Ridge regression."""
    X = filters @ basis.T  # design matrix (nfilters, nbasis)
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(X, phot)
    return ridge.coef_.T  # shape (nspec, nbasis)

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from estimated coefficients."""
    return coeffs @ basis

def main():
    rng_seed = 42
    # Wavelength grid
    wav = wavelength_grid(400, 800, 200)  # nm

    # Basis
    n_basis = 10
    basis = orthonormal_basis(n_basis, len(wav), random_state=rng_seed)

    # Filters
    n_filters = 5
    filt_width = 40.0  # nm
    filters = generate_filters(n_filters, wav, filt_width, random_state=rng_seed)

    # Synthetic spectra
    n_specs = 50
    spectra, true_coeffs = generate_spectra(
        n_specs, basis, coeff_bounds=(0, 1), noise_std=0.02, random_state=rng_seed
    )

    # Photometry
    phot = photometry_from_spectra(spectra, filters)

    # Reconstruction
    est_coeffs = reconstruct_coefficients(phot, filters, basis, alpha=0.1)
    rec_spectra = reconstruct_spectra(est_coeffs, basis)

    # Evaluation
    mse_true = np.mean((spectra - rec_spectra) ** 2)
    print(f"Mean squared error of reconstructed spectra: {mse_true:.6f}")

if __name__ == "__main__":
    main()