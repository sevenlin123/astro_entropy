import numpy as np
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# Spectral Model
# --------------------------------------------------------------------------- #

def gaussian_basis(wave, centers, widths):
    """Return Gaussian basis functions evaluated at wave."""
    return np.exp(-0.5 * ((wave - centers[:, None]) / widths[:, None]) ** 2)

def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return np.dot(coeffs, basis)

# --------------------------------------------------------------------------- #
# Photometry
# --------------------------------------------------------------------------- #

def trapezoidal_integral(flux, wave):
    """Numerical integration of flux over wavelength."""
    return np.trapz(flux, wave)

def apply_filter(spectrum, wave, filt_response):
    """Compute photometric flux for one filter."""
    return trapezoidal_integral(spectrum * filt_response, wave)

def generate_photometry(spectrum, wave, filters):
    """Compute photometry for all filters."""
    phot = []
    for filt in filters:
        phot.append(apply_filter(spectrum, wave, filt))
    return np.array(phot)

# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def build_design_matrix(filters, basis, wave):
    """Build matrix relating coefficients to photometric measurements."""
    n_filters = len(filters)
    n_basis = basis.shape[0]
    X = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            X[i, j] = trapezoidal_integral(basis[j] * filt, wave)
    return X

def reconstruct_coeffs(photometry, filters, basis, wave, alpha=0.0):
    """Reconstruct coefficients from photometry using ridge regression."""
    X = build_design_matrix(filters, basis, wave)
    clf = Ridge(alpha=alpha, fit_intercept=False)
    clf.fit(X, photometry)
    return clf.coef_

# --------------------------------------------------------------------------- #
# Example Data Generation
# --------------------------------------------------------------------------- #

def create_filters():
    """Define simple rectangular bandpasses."""
    wave = np.linspace(400, 700, 301)  # nm
    filters = []

    # Filter A: 400-500 nm
    filt_a = np.where((wave >= 400) & (wave <= 500), 1.0, 0.0)
    filters.append(filt_a)

    # Filter B: 500-600 nm
    filt_b = np.where((wave >= 500) & (wave <= 600), 1.0, 0.0)
    filters.append(filt_b)

    # Filter C: 600-700 nm
    filt_c = np.where((wave >= 600) & (wave <= 700), 1.0, 0.0)
    filters.append(filt_c)

    return wave, filters

def main():
    # Wavelength grid
    wave = np.linspace(400, 700, 301)  # nm

    # Basis functions: three Gaussians centered at 450, 550, 650 nm
    centers = np.array([450, 550, 650])
    widths = np.array([30, 30, 30])
    basis = gaussian_basis(wave, centers, widths)

    # True coefficients
    true_coeffs = np.array([1.0, 0.8, 0.5])

    # Synthesize spectrum
    spectrum = synthesize_spectrum(basis, true_coeffs)

    # Filters
    _, filters = create_filters()

    # Generate photometry
    photometry = generate_photometry(spectrum, wave, filters)

    # Reconstruct coefficients
    rec_coeffs = reconstruct_coeffs(photometry, filters, basis, wave, alpha=0.01)

    # Reconstructed spectrum
    recon_spectrum = synthesize_spectrum(basis, rec_coeffs)

    # Print results
    print("True coefficients   :", true_coeffs)
    print("Reconstructed coeffs:", rec_coeffs)
    print("Photometry          :", photometry)
    print("Reconstructed spectrum sample:", recon_spectrum[:5])

if __name__ == "__main__":
    main()