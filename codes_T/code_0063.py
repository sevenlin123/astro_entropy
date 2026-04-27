#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Define the spectral model
# -----------------------------
def create_basis(wavelengths, n_basis=5):
    """Create Gaussian basis functions centered evenly across the wavelength range."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (n_basis * 4)  # arbitrary width
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths) ** 2)
    return basis  # shape (len(wavelengths), n_basis)

def spectrum_from_coeffs(coeffs, basis):
    """Compute a spectrum as a linear combination of basis functions."""
    return basis @ coeffs  # shape (len(wavelengths),)

# -----------------------------
# 2. Generate synthetic spectra
# -----------------------------
def generate_synthetic_spectra(n_stars, basis, rng=None):
    """Generate random spectra by sampling coefficients."""
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(loc=1.0, scale=0.5, size=(n_stars, basis.shape[1]))
    spectra = coeffs @ basis.T  # shape (n_stars, len(wavelengths))
    return spectra, coeffs

# -----------------------------
# 3. Generate photometric data
# -----------------------------
def create_filter(transmission_center, transmission_width, wavelengths):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelengths - transmission_center) / transmission_width) ** 2)

def photometry_from_spectrum(spectrum, filters, wavelengths):
    """Integrate spectrum over each filter."""
    phot = []
    for filt in filters:
        # Simple trapezoidal integration
        integral = np.trapz(spectrum * filt, wavelengths)
        phot.append(integral)
    return np.array(phot)  # shape (n_filters,)

def generate_photometry(spectra, filters, wavelengths):
    """Compute photometric observations for all spectra."""
    phot_list = []
    for spec in spectra:
        phot_list.append(photometry_from_spectrum(spec, filters, wavelengths))
    return np.vstack(phot_list)  # shape (n_stars, n_filters)

# -----------------------------
# 4. Reconstruct spectra
# -----------------------------
def build_design_matrix(filters, basis, wavelengths):
    """For each filter, integrate each basis function over that filter."""
    design = []
    for filt in filters:
        integ = np.trapz(basis * filt[:, None], wavelengths, axis=0)
        design.append(integ)
    return np.stack(design, axis=0)  # shape (n_filters, n_basis)

def reconstruct_spectra(photometry, design_matrix, wavelengths):
    """Fit linear regression to recover coefficients for each star."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix.T, photometry.T)  # transposed to get shape (n_basis, n_stars)
    coeffs_hat = reg.coef_.T  # shape (n_stars, n_basis)
    # Reconstruct spectra
    reconstructed = coeffs_hat @ basis.T
    return reconstructed, coeffs_hat

# -----------------------------
# 5. Demo
# -----------------------------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(4000, 7000, 500)  # Angstroms

    # Basis functions
    basis = create_basis(wav, n_basis=5)

    # Synthetic spectra
    n_stars = 10
    rng = np.random.default_rng(seed=42)
    spectra_true, coeffs_true = generate_synthetic_spectra(n_stars, basis, rng)

    # Filters
    filt_centers = [3800, 4800, 6200]
    filt_widths  = [500, 500, 500]
    filters = [create_filter(c, w, wav) for c, w in zip(filt_centers, filt_widths)]

    # Photometric observations
    phot_obs = generate_photometry(spectra_true, filters, wav)

    # Design matrix
    design = build_design_matrix(filters, basis, wav)

    # Reconstruct spectra
    spectra_rec, coeffs_est = reconstruct_spectra(phot_obs, design, wav)

    # Evaluation
    rms_error = np.sqrt(((spectra_true - spectra_rec) ** 2).mean(axis=1))
    print("RMS error per star:", rms_error)
    print("\nTrue vs Estimated coefficients (first star):")
    print("True :", coeffs_true[0])
    print("Est. :", coeffs_est[0])