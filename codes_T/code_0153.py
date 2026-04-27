import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

def build_basis_spectra(num_bases, num_wave):
    """Generate synthetic basis spectra (Gaussian peaks)."""
    wavelengths = np.linspace(400, 1000, num_wave)
    basis = []
    for i in range(num_bases):
        center = np.random.uniform(450, 950)
        width = np.random.uniform(20, 60)
        peak = gaussian(num_wave, std=width / (wavelengths[1] - wavelengths[0]))
        peak = np.roll(peak, int((center - wavelengths[0]) / (wavelengths[1]-wavelengths[0])))
        basis.append(peak)
    return np.array(basis), wavelengths

def generate_random_coeffs(num_bases, low=0, high=1):
    """Random coefficients for linear combination."""
    return np.random.uniform(low, high, size=num_bases)

def combine_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return np.dot(coeffs, basis)

def create_filters(num_filters, num_wave):
    """Generate simple Gaussian bandpass filters."""
    wavelengths = np.linspace(400, 1000, num_wave)
    filters = []
    for _ in range(num_filters):
        center = np.random.uniform(450, 950)
        width = np.random.uniform(30, 80)
        filt = gaussian(num_wave, std=width / (wavelengths[1] - wavelengths[0]))
        filt = np.roll(filt, int((center - wavelengths[0]) / (wavelengths[1]-wavelengths[0])))
        filt /= filt.max()
        filters.append(filt)
    return np.array(filters), wavelengths

def compute_photometry(spectrum, filters):
    """Integrate spectrum over each filter."""
    return np.dot(filters, spectrum)

def reconstruct_spectrum(photometry, filters, basis):
    """Recover spectrum coefficients using ridge regression."""
    # Solve for coeffs that best reproduce photometry
    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(filters, photometry)
    coeffs_est = reg.coef_
    reconstructed = np.dot(coeffs_est, basis)
    return reconstructed, coeffs_est

def main():
    np.random.seed(42)
    num_bases = 5
    num_wave = 300
    num_filters = 4

    basis, wavelengths = build_basis_spectra(num_bases, num_wave)
    true_coeffs = generate_random_coeffs(num_bases)
    true_spectrum = combine_spectrum(basis, true_coeffs)

    filters, _ = create_filters(num_filters, num_wave)
    photometry = compute_photometry(true_spectrum, filters)

    recon_spectrum, est_coeffs = reconstruct_spectrum(photometry, filters, basis)

    print("True coefficients:     ", true_coeffs)
    print("Estimated coefficients:", est_coeffs)
    print("Mean squared error:", np.mean((true_spectrum - recon_spectrum)**2))

if __name__ == "__main__":
    main()