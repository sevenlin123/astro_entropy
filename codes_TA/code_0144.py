import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(start=400, stop=800, step=1):
    """Wavelength grid in nanometers."""
    return np.arange(start, stop + step, step)

def gaussian_peak(center, width, wavelengths):
    """Generate a Gaussian peak over the wavelength grid."""
    sigma = width / 2.3548  # FWHM to sigma
    peak = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    return peak / peak.max()  # normalize to unity

def create_basis(num_basis, wavelengths, rng=np.random.default_rng()):
    """Create a set of Gaussian basis spectra."""
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=num_basis)
    widths = rng.uniform(15, 40, size=num_basis)  # in nm
    basis = np.vstack([gaussian_peak(c, w, wavelengths) for c, w in zip(centers, widths)])
    return basis

def generate_synthetic_spectra(num_spectra, basis, rng=np.random.default_rng()):
    """Generate synthetic spectra as linear combinations of the basis."""
    coeffs = rng.uniform(0.0, 1.0, size=(num_spectra, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs

def create_filters(num_filters, wavelengths, rng=np.random.default_rng()):
    """Define a set of Gaussian photometric filters."""
    centers = rng.uniform(wavelengths[0]+20, wavelengths[-1]-20, size=num_filters)
    widths = rng.uniform(30, 50, size=num_filters)  # in nm
    filters = np.vstack([gaussian_peak(c, w, wavelengths) for c, w in zip(centers, widths)])
    # Normalize to unit area for photometric consistency
    filters /= filters.sum(axis=1, keepdims=True)
    return filters

def compute_photometry(spectra, filters):
    """Compute synthetic photometric measurements."""
    return spectra @ filters.T  # shape: (num_spectra, num_filters)

def fit_reconstruction_model(photometry, coeffs):
    """Fit a linear model mapping photometry to basis coefficients."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(photometry, coeffs)
    return reg

def reconstruct_spectrum(reg, photometry_sample, basis):
    """Reconstruct a spectrum from a single photometric measurement."""
    coeffs_pred = reg.predict(photometry_sample.reshape(1, -1))
    reconstructed = coeffs_pred @ basis
    return reconstructed.squeeze(), coeffs_pred.squeeze()

def main():
    rng = np.random.default_rng(seed=42)
    wav = create_wavelength_grid()
    
    # 1. Spectral model (basis functions)
    basis = create_basis(num_basis=5, wavelengths=wav, rng=rng)
    
    # 2. Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(num_spectra=200, basis=basis, rng=rng)
    
    # 3. Generate photometric data
    filters = create_filters(num_filters=4, wavelengths=wav, rng=rng)
    photometry = compute_photometry(spectra, filters)
    
    # Train reconstruction model
    reg = fit_reconstruction_model(photometry, true_coeffs)
    
    # 4. Reconstruct a new synthetic spectrum
    new_spectrum, new_coeffs = generate_synthetic_spectra(num_spectra=1, basis=basis, rng=rng)
    new_photometry = compute_photometry(new_spectrum, filters)
    
    rec_spec, rec_coeffs = reconstruct_spectrum(reg, new_photometry, basis)
    
    # Display error metrics
    spec_err = np.linalg.norm(rec_spec - new_spectrum.squeeze()) / np.linalg.norm(new_spectrum.squeeze())
    coeff_err = np.linalg.norm(rec_coeffs - new_coeffs.squeeze()) / np.linalg.norm(new_coeffs.squeeze())
    print(f"Spectral reconstruction relative L2 error: {spec_err:.4f}")
    print(f"Coefficient reconstruction relative L2 error: {coeff_err:.4f}")

if __name__ == "__main__":
    main()