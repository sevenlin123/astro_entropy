import numpy as np
from scipy.integrate import simps
from sklearn.metrics import mean_squared_error

def generate_basis(num_bases=5, num_wavelengths=200):
    """Generate random basis spectra."""
    rng = np.random.default_rng(0)
    wavelengths = np.linspace(400, 800, num_wavelengths)  # nm
    basis = rng.standard_normal((num_wavelengths, num_bases))
    # Normalize each basis to unit area
    basis /= simps(basis, wavelengths, axis=0)
    return wavelengths, basis

def generate_synthetic_spectra(basis, n_samples=50):
    """Generate synthetic spectra as linear combos of basis."""
    rng = np.random.default_rng(1)
    coeffs = rng.uniform(0.1, 1.0, size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

def gaussian_filter(wavelengths, center, width):
    """Return Gaussian filter response."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths, num_filters=4):
    """Create simple Gaussian filters."""
    rng = np.random.default_rng(2)
    centers = rng.uniform(wavelengths.min() + 50, wavelengths.max() - 50,
                          size=num_filters)
    widths = np.full(num_filters, 30.0)  # nm
    filters = [gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)  # shape (num_filters, num_wavelengths)

def photometry_from_spectrum(spectrum, wavelengths, filters):
    """Compute photometric fluxes by integrating spectrum * filter."""
    fluxes = []
    for filt in filters:
        flux = simps(spectrum * filt, wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)

def reconstruct_spectrum_from_photometry(photometry, basis, filters, wavelengths):
    """
    Reconstruct spectrum from photometry by solving linear least-squares
    for coefficients and then synthesizing the spectrum.
    """
    # Precompute response matrix A_{fk} = ∫ basis_k * filter_f
    A = np.array([simps(basis * filt, wavelengths, axis=0) for filt in filters])
    coeffs_est = np.linalg.lstsq(A.T, photometry, rcond=None)[0]  # shape (n_bases,)
    spectrum_est = coeffs_est @ basis.T
    return spectrum_est, coeffs_est

def main():
    wavelengths, basis = generate_basis()
    spectra, coeffs_true = generate_synthetic_spectra(basis, n_samples=20)
    filters = generate_filters(wavelengths)
    # Compute photometry for each synthetic spectrum
    photometry = np.array([photometry_from_spectrum(s, wavelengths, filters)
                           for s in spectra])  # shape (n_samples, n_filters)
    # Reconstruct each spectrum
    reconstructions = []
    coeffs_est_all = []
    for i in range(spectra.shape[0]):
        spec_est, coeff_est = reconstruct_spectrum_from_photometry(
            photometry[i], basis, filters, wavelengths)
        reconstructions.append(spec_est)
        coeffs_est_all.append(coeff_est)
    reconstructions = np.array(reconstructions)
    # Evaluate reconstruction error
    mse_per_sample = mean_squared_error(spectra, reconstructions, multioutput='raw_values')
    print("Mean squared error per sample (first 5):", mse_per_sample[:5])
    print("Average MSE:", np.mean(mse_per_sample))

if __name__ == "__main__":
    main()