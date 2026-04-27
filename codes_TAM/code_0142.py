#!/usr/bin/env python3
import numpy as np


def generate_basis(n_wavelengths, n_coeffs, seed=0):
    """Create an orthonormal basis of spectral components."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(n_wavelengths, n_coeffs))
    q, _ = np.linalg.qr(raw)          # orthonormalise
    return q


def generate_synthetic_spectra(n_samples, basis, coeff_std=1.0, seed=1):
    """Generate random spectra as linear combinations of the basis."""
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal(size=(n_samples, basis.shape[1])) * coeff_std
    spectra = coeffs @ basis.T       # (n_samples, n_wavelengths)
    return spectra, coeffs


def generate_filters(n_filters, n_wavelengths, seed=2):
    """Produce simple top‑hat photometric filter responses."""
    rng = np.random.default_rng(seed)
    filt = np.zeros((n_filters, n_wavelengths))
    widths = rng.integers(low=5, high=n_wavelengths // 2, size=n_filters)
    starts = rng.integers(low=0, high=n_wavelengths - widths, size=n_filters)
    for i in range(n_filters):
        filt[i, starts[i] : starts[i] + widths[i]] = 1.0
    return filt


def compute_photometry(spectra, filters, dwl=1.0):
    """Integrate spectra through the filter responses."""
    return spectra @ (filters.T * dwl)   # (n_samples, n_filters)


def reconstruct_coeffs(photometry, filters, basis):
    """Infer spectral coefficients from photometry."""
    # Build matrix A where A_{i,j} = ∫ filter_i(λ) * basis_j(λ) dλ
    A = filters @ basis                 # (n_filters, n_coeffs)
    coeffs_rec = []
    for p in photometry:
        coeffs_rec.append(np.linalg.lstsq(A, p, rcond=None)[0])
    return np.vstack(coeffs_rec)        # (n_samples, n_coeffs)


def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from estimated coefficients."""
    return coeffs @ basis.T             # (n_samples, n_wavelengths)


def main():
    n_wl = 200           # number of wavelength samples
    n_coeff = 12         # number of basis components
    n_samples = 8        # number of synthetic stars
    n_filters = 5        # number of photometric bands
    dwl = 1.0            # wavelength step (arbitrary units)

    # Core simulation
    basis = generate_basis(n_wl, n_coeff)
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis)
    filters = generate_filters(n_filters, n_wl)
    photometry = compute_photometry(spectra_true, filters, dwl)
    coeffs_est = reconstruct_coeffs(photometry, filters, basis)
    spectra_est = reconstruct_spectra(coeffs_est, basis)

    # Simple diagnostics
    print("Coefficient reconstruction error (RMSE):",
          np.sqrt(((coeffs_true - coeffs_est) ** 2).mean()))
    print("Spectrum reconstruction RMSE:",
          np.sqrt(((spectra_true - spectra_est) ** 2).mean()))


if __name__ == "__main__":
    main()