#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral Model ----------
def generate_basis_spectra(n_basis, wavelengths):
    """Generate simple Gaussian basis spectra."""
    basis = []
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (2 * n_basis)
    for c in centers:
        gauss = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(gauss)
    return np.array(basis)  # shape (n_basis, n_wavelength)

# ---------- Synthetic Spectra ----------
def generate_synthetic_spectra(n_samples, basis, rng):
    """Generate spectra as random linear combinations of basis spectra."""
    n_basis, n_wav = basis.shape
    coeffs = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_basis))
    spectra = coeffs @ basis  # shape (n_samples, n_wav)
    return spectra, coeffs

# ---------- Filters ----------
def generate_filters():
    """Define simple top‑hat filters (UBVRI)."""
    # Simple wavelength grid for filters
    w_min, w_max = 350, 900  # nm
    filt_centers = [360, 440, 550, 660, 770]
    filt_widths  = [50, 50, 60, 60, 70]
    wavelengths = np.linspace(300, 1000, 400)  # nm
    filters = {}
    for name, cen, wid in zip(['U', 'B', 'V', 'R', 'I'], filt_centers, filt_widths):
        trans = np.where((wavelengths >= cen - wid/2) & (wavelengths <= cen + wid/2), 1.0, 0.0)
        filters[name] = trans
    return filters, wavelengths

# ---------- Photometry ----------
def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra through each filter."""
    phot = []
    for filt in filters.values():
        # Normalize by filter width to avoid scaling issues
        integ = np.trapz(spectra * filt, wavelengths, axis=1)
        phot.append(integ)
    return np.vstack(phot).T  # shape (n_samples, n_filters)

# ---------- Reconstruction ----------
def reconstruct_spectra(photometry, filters, basis, wavelengths):
    """Recover spectra from photometry using linear regression on basis integrals."""
    # Precompute integrals of each basis through each filter
    n_filters = len(filters)
    n_basis, n_wav = basis.shape
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters.values()):
        A[i] = np.trapz(basis.T * filt, wavelengths, axis=1)
    recon = []
    for phot in photometry:
        reg = LinearRegression(fit_intercept=False).fit(A, phot)
        coeffs = reg.coef_
        spec = coeffs @ basis
        recon.append(spec)
    return np.array(recon)  # shape (n_samples, n_wavelength)

# ---------- Main ----------
def main():
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wavelengths = np.linspace(300, 1000, 500)  # nm

    # Basis spectra
    basis = generate_basis_spectra(n_basis=5, wavelengths=wavelengths)

    # Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(n_samples=10, basis=basis, rng=rng)

    # Filters
    filters, filt_wavelengths = generate_filters()

    # Compute photometry
    photometry = compute_photometry(spectra, filters, filt_wavelengths)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(photometry, filters, basis, filt_wavelengths)

    # Evaluation
    mse = np.mean((spectra - recon_spectra)**2, axis=1)
    for i, val in enumerate(mse):
        print(f"Sample {i+1}: MSE between true and reconstructed spectrum = {val:.6e}")

if __name__ == "__main__":
    main()