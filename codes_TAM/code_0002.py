#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------
# 1. Spectral model definition
# ------------------------------
def create_basis_functions(n_basis, w_min, w_max, n_wavelength):
    """Generate a set of Gaussian basis functions."""
    wavelengths = np.linspace(w_min, w_max, n_wavelength)
    centers = np.linspace(w_min, w_max, n_basis)
    widths = (w_max - w_min) / (2 * n_basis)

    basis = []
    for c in centers:
        gauss = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(gauss)
    return np.array(basis)  # shape: (n_basis, n_wavelength)

# ------------------------------
# 2. Synthetic spectra generation
# ------------------------------
def generate_synthetic_spectrum(coeffs, basis):
    """Combine basis functions with given coefficients."""
    return coeffs @ basis  # shape: (n_wavelength,)

# ------------------------------
# 3. Photometric data simulation
# ------------------------------
def create_filter_curve(wavelength, center, width, shape="top_hat"):
    """Create a simple filter transmission curve."""
    if shape == "top_hat":
        trans = np.where(
            (wavelength >= center - width / 2) & (wavelength <= center + width / 2),
            1.0,
            0.0,
        )
    else:
        # Gaussian filter
        trans = np.exp(-0.5 * ((wavelength - center) / width) ** 2)
    return trans

def compute_photometry(spectrum, wavelengths, filters):
    """Compute synthetic photometry for each filter."""
    photometry = []
    for filt in filters:
        integrand = spectrum * filt
        flux = np.trapz(integrand, wavelengths)
        norm = np.trapz(filt, wavelengths)
        photometry.append(flux / norm if norm > 0 else 0)
    return np.array(photometry)

# ------------------------------
# 4. Spectrum reconstruction
# ------------------------------
def construct_design_matrix(basis, filters, wavelengths):
    """Build the matrix that maps coefficients to photometry."""
    design = []
    for filt in filters:
        col = []
        for b in basis:
            integrand = b * filt
            val = np.trapz(integrand, wavelengths)
            norm = np.trapz(filt, wavelengths)
            col.append(val / norm if norm > 0 else 0)
        design.append(col)
    return np.array(design)  # shape: (n_filters, n_basis)

def reconstruct_spectrum(photometry, design_matrix, basis):
    """Reconstruct spectrum from photometry using Ridge regression."""
    reg = Ridge(alpha=1e-6, fit_intercept=False)
    coeffs = reg.fit(design_matrix, photometry).coef_
    return coeffs @ basis

# ------------------------------
# 5. Main routine
# ------------------------------
def main():
    # Wavelength grid
    w_min, w_max = 300.0, 1000.0  # nm
    n_wavelength = 500
    wavelengths = np.linspace(w_min, w_max, n_wavelength)

    # Basis functions
    n_basis = 10
    basis = create_basis_functions(n_basis, w_min, w_max, n_wavelength)

    # Synthetic spectrum coefficients (random)
    np.random.seed(42)
    true_coeffs = np.random.randn(n_basis)
    true_spectrum = generate_synthetic_spectrum(true_coeffs, basis)

    # Define filters
    filter_centers = [350, 450, 550, 650, 750]  # nm
    filter_widths = [50, 50, 50, 50, 50]
    filters = [
        create_filter_curve(wavelengths, c, w) for c, w in zip(filter_centers, filter_widths)
    ]

    # Simulate photometry
    photometry = compute_photometry(true_spectrum, wavelengths, filters)

    # Reconstruction
    design = construct_design_matrix(basis, filters, wavelengths)
    recovered_spectrum = reconstruct_spectrum(photometry, design, basis)

    # Output results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients (approximate):\n", 
          np.linalg.lstsq(design, photometry, rcond=None)[0])
    print("\nMean squared error between true and recovered spectrum:",
          np.mean((true_spectrum - recovered_spectrum) ** 2))

if __name__ == "__main__":
    main()