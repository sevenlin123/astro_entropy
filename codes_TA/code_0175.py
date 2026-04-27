#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def gaussian(x, mu, sigma):
    """Simple Gaussian profile."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def basis_spectra(wl, n_basis=5):
    """
    Generate a set of orthogonal basis spectra (simple Gaussians) over wavelength array `wl`.
    Returns an array of shape (n_basis, len(wl)).
    """
    mus = np.linspace(np.min(wl), np.max(wl), n_basis)
    sigmas = (np.max(wl) - np.min(wl)) / (4 * n_basis)
    basis = np.array([gaussian(wl, mu, sigmas) for mu in mus])
    # Normalize each basis spectrum
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis

def synthetic_spectrum(coeffs, basis, wl):
    """Linear combination of basis spectra with given coefficients."""
    return np.dot(coeffs, basis)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_example():
    # Wavelength grid (in nm)
    wl = np.linspace(300, 800, 500)  # 300–800 nm

    # Basis spectra
    basis = basis_spectra(wl, n_basis=5)

    # True coefficients for a synthetic star (random but within [0,1])
    true_coeffs = np.random.rand(basis.shape[0])

    # Synthetic spectrum
    spec = synthetic_spectrum(true_coeffs, basis, wl)
    return wl, spec, basis, true_coeffs

# ----------------------------------------------------------------------
# 3. Generate photometric data
# ----------------------------------------------------------------------
def filter_response(wl, center, width):
    """Gaussian filter transmission curve."""
    return gaussian(wl, center, width)

def photometric_bands():
    """
    Define a few broad-band filters (center wavelength in nm, width in nm).
    Example: Johnson UBV
    """
    return {
        'U': (365, 60),
        'B': (445, 90),
        'V': (551, 90),
    }

def compute_photometric_fluxes(spec, wl, bands):
    """
    Compute integrated flux through each band: integral(spec * response) / integral(response).
    Returns a dict of band fluxes.
    """
    fluxes = {}
    for name, (cen, wid) in bands.items():
        resp = filter_response(wl, cen, wid)
        flux = simps(spec * resp, wl) / simps(resp, wl)
        fluxes[name] = flux
    return fluxes

# ----------------------------------------------------------------------
# 4. Reconstruct spectrum from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum(fluxes, bands, wl, basis):
    """
    Reconstruct the spectrum by solving for coefficients that best fit the photometric fluxes.
    Uses linear least squares on the band-integrated responses.
    Returns the reconstructed spectrum and recovered coefficients.
    """
    # Build design matrix: for each band, integrate basis * filter
    X = []
    for name, (cen, wid) in bands.items():
        resp = filter_response(wl, cen, wid)
        row = np.array([simps(b * resp, wl) / simps(resp, wl) for b in basis])
        X.append(row)
    X = np.vstack(X)  # shape (n_bands, n_basis)

    y = np.array([fluxes[name] for name in bands.keys()])  # observed fluxes

    # Least-squares solve for coefficients
    lr = LinearRegression(fit_intercept=False).fit(X, y)
    coeffs_recon = lr.coef_

    # Reconstruct spectrum
    recon_spec = synthetic_spectrum(coeffs_recon, basis, wl)
    return recon_spec, coeffs_recon

# ----------------------------------------------------------------------
# 5. Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Generate example
    wl, true_spec, basis, true_coeffs = generate_example()

    # Compute photometry
    bands = photometric_bands()
    fluxes = compute_photometric_fluxes(true_spec, wl, bands)

    # Reconstruct
    recon_spec, rec_coeffs = reconstruct_spectrum(fluxes, bands, wl, basis)

    # Output results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", rec_coeffs)
    print("\nDifference in coefficients:\n", true_coeffs - rec_coeffs)
    print("\nReconstruction error (L2 norm):", np.linalg.norm(true_spec - recon_spec))