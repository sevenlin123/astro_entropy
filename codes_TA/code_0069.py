import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Spectral model: a set of basis spectra (e.g. Gaussian features)
# ----------------------------------------------------------------------
def make_wavelength_grid(n_points=100, lam_min=400, lam_max=800):
    """Return a monotonically increasing wavelength grid."""
    return np.linspace(lam_min, lam_max, n_points)

def gaussian_basis(n_basis, wavelengths):
    """Generate a set of Gaussian basis spectra."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (4 * n_basis)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, n_wavelength)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def synthesize_spectrum(coeffs, basis):
    """
    Construct a synthetic spectrum as a linear combination of basis
    spectra weighted by coeffs.
    """
    return coeffs @ basis  # shape (n_wavelength,)

# ----------------------------------------------------------------------
# 3. Define filter transmission curves
# ----------------------------------------------------------------------
def gaussian_filter_response(n_filters, wavelengths):
    """Return a set of Gaussian filter responses."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    widths = (wavelengths[-1] - wavelengths[0]) / (6 * n_filters)
    filters = []
    for c in centers:
        f = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        filters.append(f)
    return np.array(filters)  # shape (n_filters, n_wavelength)

# ----------------------------------------------------------------------
# 4. Compute photometric fluxes by integrating spectrum over filters
# ----------------------------------------------------------------------
def compute_photometry(spectrum, filters, wavelengths):
    """
    Integrate spectrum over each filter using trapezoidal rule.
    Returns an array of photometric fluxes (one per filter).
    """
    # Normalize filter transmissions to unit integral for consistency
    norm_filters = filters / np.trapz(filters, wavelengths, axis=1, keepdims=True)
    return np.trapz(spectrum * norm_filters.T, wavelengths, axis=1)

# ----------------------------------------------------------------------
# 5. Reconstruct spectrum from photometric data
# ----------------------------------------------------------------------
def reconstruct_spectrum(photometry, filters, basis, wavelengths):
    """
    Recover the coefficient vector that best reproduces the given
    photometry. Uses Ridge regression for stability.
    """
    # Design matrix A maps coefficients -> photometry
    # A_ij = integral(filter_j * basis_i)
    A = np.trapz(basis.T * filters.T, wavelengths, axis=1).T
    reg = Ridge(alpha=1.0, fit_intercept=False, solver='auto')
    reg.fit(A, photometry)
    coeffs_rec = reg.coef_
    return synthesize_spectrum(coeffs_rec, basis)

# ----------------------------------------------------------------------
# 6. Example workflow
# ----------------------------------------------------------------------
def main():
    # Setup
    wavelengths = make_wavelength_grid()
    n_basis = 5
    n_filters = 3

    # Basis spectra
    basis = gaussian_basis(n_basis, wavelengths)

    # Generate a random true spectrum
    np.random.seed(42)
    true_coeffs = np.random.randn(n_basis)
    true_spectrum = synthesize_spectrum(true_coeffs, basis)

    # Filters
    filters = gaussian_filter_response(n_filters, wavelengths)

    # Photometric observations
    photometry = compute_photometry(true_spectrum, filters, wavelengths)

    # Reconstruction
    recovered_spectrum = reconstruct_spectrum(photometry, filters, basis, wavelengths)

    # Evaluate
    print("True coefficients:      ", true_coeffs)
    print("Recovered coefficients:", reg.coef_)
    print("\nReconstruction error (RMSE):",
          np.sqrt(np.mean((true_spectrum - recovered_spectrum)**2)))

if __name__ == "__main__":
    main()