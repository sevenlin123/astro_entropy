import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

def create_wavelength_grid(wl_min=300, wl_max=1000, num=1000):
    """Uniform wavelength grid."""
    return np.linspace(wl_min, wl_max, num)

def gaussian_basis(wavelength, centers, widths, heights):
    """Return a matrix of Gaussian basis spectra."""
    n_basis = len(centers)
    mat = np.zeros((len(wavelength), n_basis))
    for i, (c, w, h) in enumerate(zip(centers, widths, heights)):
        mat[:, i] = h * np.exp(-0.5 * ((wavelength - c) / w) ** 2)
    return mat

def generate_synthetic_spectra(basis_matrix, n_spectra, coeff_range=(0.5, 1.5)):
    """Generate synthetic spectra as random linear combinations of basis spectra."""
    n_basis = basis_matrix.shape[1]
    coeffs = np.random.uniform(coeff_range[0], coeff_range[1],
                               size=(n_spectra, n_basis))
    spectra = coeffs @ basis_matrix.T
    return spectra, coeffs

def gaussian_filter_curve(wavelength, center, width):
    """Simple Gaussian transmission curve."""
    return np.exp(-0.5 * ((wavelength - center) / width) ** 2)

def generate_filters(wavelength, n_filters=3):
    """Generate a set of synthetic photometric filters."""
    centers = np.linspace(400, 800, n_filters)
    widths = np.full(n_filters, 50.0)
    filters = np.array([gaussian_filter_curve(wavelength, c, w)
                        for c, w in zip(centers, widths)])
    return filters, centers, widths

def integrate_over_filter(spectrum, filter_curve, wavelength):
    """Compute photon-count weighted flux through a filter."""
    # Assume flat photon response; integrate spectrum * filter / ∫filter
    numerator = simps(spectrum * filter_curve, wavelength)
    denom = simps(filter_curve, wavelength)
    return numerator / denom

def compute_photometry(spectra, filters, wavelength):
    """Calculate photometric measurements for all spectra."""
    n_specs = spectra.shape[0]
    n_filt = filters.shape[0]
    photometry = np.zeros((n_specs, n_filt))
    for j, filt in enumerate(filters):
        photometry[:, j] = [integrate_over_filter(sp, filt, wavelength)
                            for sp in spectra]
    return photometry

def reconstruct_spectra(photometry, basis_matrix, filters, alpha=1.0):
    """
    Reconstruct spectra from photometry.
    Uses ridge regression to fit coefficients for basis spectra.
    """
    # Build design matrix: for each filter, sum over basis contributions
    # design[i, k] = ∫ basis_k * filter_i / ∫ filter_i
    n_filt = filters.shape[0]
    n_basis = basis_matrix.shape[1]
    design = np.zeros((n_filt, n_basis))
    for i, filt in enumerate(filters):
        for k in range(n_basis):
            design[i, k] = simps(basis_matrix[:, k] * filt,
                                 axis=0) / simps(filt, axis=0)
    # Fit ridge regression per spectrum
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design, photometry.T)  # transpose because sklearn expects samples x features
    coeffs_rec = ridge.coef_.T   # shape (n_specs, n_basis)
    reconstructed = coeffs_rec @ basis_matrix.T
    return reconstructed, coeffs_rec

def main():
    # 1. Define wavelength grid
    wavelength = create_wavelength_grid()

    # 2. Define basis spectra (Gaussian peaks)
    n_basis = 5
    centers = np.linspace(350, 950, n_basis)
    widths = np.full(n_basis, 30.0)
    heights = np.ones(n_basis)
    basis_matrix = gaussian_basis(wavelength, centers, widths, heights)

    # 3. Generate synthetic spectra
    n_spectra = 10
    spectra, true_coeffs = generate_synthetic_spectra(basis_matrix, n_spectra)

    # 4. Generate photometric filters
    filters, filt_centers, filt_widths = generate_filters(wavelength)

    # 5. Compute photometry
    photometry = compute_photometry(spectra, filters, wavelength)

    # 6. Reconstruct spectra from photometry
    reconstructed, rec_coeffs = reconstruct_spectra(photometry, basis_matrix,
                                                    filters, alpha=1.0)

    # 7. Compare true vs reconstructed spectra (print RMS error)
    rms_error = np.sqrt(np.mean((spectra - reconstructed)**2, axis=1))
    for idx, err in enumerate(rms_error):
        print(f"Spectrum {idx}: RMS reconstruction error = {err:.4f}")

if __name__ == "__main__":
    main()