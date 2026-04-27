#!/usr/bin/env python3
import numpy as np
from scipy.integrate import trapz

# ------------------ Utility functions ------------------
def wavelength_grid(start_nm=300, end_nm=2500, n_points=1000):
    """Create a linear wavelength grid in nanometers."""
    return np.linspace(start_nm, end_nm, n_points)

def gaussian(x, mu, sigma):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

# ------------------ Spectral model --------------------
def generate_basis_spectra(wavelengths, n_basis=5, seed=42):
    """
    Generate a set of simple basis spectra:
    random combinations of Gaussians.
    """
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(n_basis):
        amp = rng.uniform(0.5, 1.5)
        mu = rng.uniform(wavelengths[0], wavelengths[-1])
        sigma = rng.uniform(50, 200)
        spec = amp * gaussian(wavelengths, mu, sigma)
        basis.append(spec)
    return np.array(basis)  # shape: (n_basis, n_wave)

def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return basis.T @ coeffs  # shape: (n_wave,)

# ------------------ Filter generation ------------------
def generate_filters(wavelengths, n_filters=3, seed=24):
    """
    Create simple Gaussian transmission curves for photometric filters.
    """
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        mu = rng.uniform(wavelengths[0], wavelengths[-1])
        sigma = rng.uniform(30, 150)
        trans = gaussian(wavelengths, mu, sigma)
        filters.append(trans)
    return np.array(filters)  # shape: (n_filters, n_wave)

# ------------------ Photometry ------------------------
def compute_photometry(spectrum, wavelengths, filters):
    """
    Integrate spectrum times each filter transmission.
    Returns array of fluxes.
    """
    phot = []
    for filt in filters:
        flux = trapz(spectrum * filt, wavelengths)
        phot.append(flux)
    return np.array(phot)

def add_noise(fluxes, sigma_frac=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noisy = fluxes + rng.normal(scale=sigma_frac * fluxes, size=fluxes.shape)
    return noisy

# ------------------ Reconstruction ---------------------
def compute_filter_matrix(basis, filters, wavelengths):
    """
    Build matrix M where M[i, j] = integral of basis[j] * filter[i].
    """
    n_filters, n_wave = filters.shape
    n_basis = basis.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            M[i, j] = trapz(basis[j] * filters[i], wavelengths)
    return M

def reconstruct_coefficients(photometry, filter_matrix):
    """
    Solve least-squares problem M * coeffs = photometry.
    """
    coeffs, *_ = np.linalg.lstsq(filter_matrix, photometry, rcond=None)
    return coeffs

# ------------------ Main workflow ----------------------
def main():
    # Wavelength grid
    wav = wavelength_grid()

    # Basis spectra
    basis = generate_basis_spectra(wav, n_basis=5)

    # True coefficients
    true_coeffs = np.array([1.0, 0.8, 0.5, 0.2, 0.1])

    # Synthetic spectrum
    true_spectrum = synthesize_spectrum(basis, true_coeffs)

    # Filters
    filters = generate_filters(wav, n_filters=3)

    # Photometric measurements
    true_flux = compute_photometry(true_spectrum, wav, filters)
    noisy_flux = add_noise(true_flux, sigma_frac=0.05, seed=123)

    # Reconstruction
    M = compute_filter_matrix(basis, filters, wav)
    rec_coeffs = reconstruct_coefficients(noisy_flux, M)
    recon_spectrum = synthesize_spectrum(basis, rec_coeffs)

    # Simple diagnostics
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs  :", rec_coeffs.round(3))
    diff = recon_spectrum - true_spectrum
    print(f"Reconstruction RMS error: {np.sqrt(np.mean(diff**2)):.4f}")

if __name__ == "__main__":
    main()