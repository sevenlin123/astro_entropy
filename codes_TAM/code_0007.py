import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge


def gaussian_basis(wavelengths, center, width):
    """Return a Gaussian basis function evaluated at given wavelengths."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def create_basis_functions(wavelengths, centers, widths):
    """Create a list of basis functions."""
    return [gaussian_basis(wavelengths, c, w) for c, w in zip(centers, widths)]


def synthesize_spectrum(basis_funcs, weights, noise_level=0.0):
    """Linear combination of basis functions with optional Gaussian noise."""
    spectrum = np.zeros_like(basis_funcs[0])
    for bf, w in zip(basis_funcs, weights):
        spectrum += w * bf
    if noise_level > 0:
        spectrum += np.random.normal(scale=noise_level, size=spectrum.shape)
    return spectrum


def create_filter_transmission(wavelengths, center, width):
    """Gaussian filter transmission curve."""
    return gaussian_basis(wavelengths, center, width)


def create_filters(wavelengths, centers, widths):
    """Generate filter transmission curves."""
    return [create_filter_transmission(wavelengths, c, w) for c, w in zip(centers, widths)]


def compute_photometry(spectrum, filters):
    """Integrate spectrum through each filter (simple weighted sum)."""
    return np.array([simps(spectrum * filt, x=None) for filt in filters])


def build_design_matrix(filters, basis_funcs):
    """Matrix of integrals of each basis function through each filter."""
    mat = np.empty((len(filters), len(basis_funcs)))
    for i, filt in enumerate(filters):
        for j, bf in enumerate(basis_funcs):
            mat[i, j] = simps(bf * filt, x=None)
    return mat


def reconstruct_coefficients(design_matrix, photometry, alpha=1e-4):
    """Solve for basis coefficients via ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(design_matrix, photometry)
    return ridge.coef_


def reconstruct_spectrum(basis_funcs, coeffs):
    """Sum basis functions weighted by reconstructed coefficients."""
    spectrum = np.zeros_like(basis_funcs[0])
    for bf, c in zip(basis_funcs, coeffs):
        spectrum += c * bf
    return spectrum


def main():
    # Wavelength grid
    wl = np.linspace(400, 700, 301)  # nm

    # Basis function definition
    basis_centers = [450, 500, 550, 600, 650]
    basis_widths = [20, 20, 20, 20, 20]
    basis_funcs = create_basis_functions(wl, basis_centers, basis_widths)

    # True coefficients for synthetic spectrum
    true_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    # Generate synthetic spectrum
    spectrum_true = synthesize_spectrum(basis_funcs, true_weights, noise_level=0.01)

    # Filters definition
    filter_centers = [425, 475, 525, 575, 625]
    filter_widths = [30, 30, 30, 30, 30]
    filters = create_filters(wl, filter_centers, filter_widths)

    # Compute photometry from true spectrum
    photometry = compute_photometry(spectrum_true, filters)

    # Build design matrix
    design_mat = build_design_matrix(filters, basis_funcs)

    # Reconstruct coefficients from photometry
    recon_weights = reconstruct_coefficients(design_mat, photometry, alpha=1e-3)

    # Reconstruct spectrum
    spectrum_recon = reconstruct_spectrum(basis_funcs, recon_weights)

    # Print results
    print("True weights:      ", true_weights)
    print("Reconstructed weights:", recon_weights)
    print("\nPhotometry:")
    print(photometry)
    print("\nDifference between true and reconstructed spectrum (RMS):",
          np.sqrt(np.mean((spectrum_true - spectrum_recon) ** 2)))


if __name__ == "__main__":
    main()