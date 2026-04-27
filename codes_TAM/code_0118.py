import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# 1. Spectral model: basis functions (simple Gaussians)
# -------------------------------------------------------------
def spectral_basis(wavelengths, centers=[450, 550, 650], widths=[30, 40, 35]):
    """Generate basis spectra."""
    basis = []
    for c, w in zip(centers, widths):
        spectrum = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        basis.append(spectrum)
    return np.vstack(basis)  # shape (n_basis, n_wavelengths)

# -------------------------------------------------------------
# 2. Generate synthetic spectra
# -------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, n_basis=3, noise_std=0.02):
    """Generate synthetic spectra as linear combos of basis."""
    basis = spectral_basis(wavelengths)[:n_basis]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = coeffs @ basis          # shape (n_samples, n_wavelengths)
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

# -------------------------------------------------------------
# 3. Photometric filter transmission curves
# -------------------------------------------------------------
def filter_transmissions(wavelengths, filter_names=['U','B','V','R','I']):
    """Create simple Gaussian filter curves."""
    trans = {}
    params = {
        'U': (350, 50),
        'B': (440, 40),
        'V': (550, 30),
        'R': (660, 30),
        'I': (790, 50)
    }
    for f in filter_names:
        center, width = params[f]
        trans[f] = np.exp(-0.5 * ((wavelengths - center) / width)**2)
    return trans

# -------------------------------------------------------------
# 4. Generate photometric data from spectra
# -------------------------------------------------------------
def compute_photometry(spectra, wavelengths, filters):
    """Integrate spectra over filter transmissions to get fluxes."""
    n_filters = len(filters)
    n_samples = spectra.shape[0]
    photometry = np.empty((n_samples, n_filters))
    for i, fname in enumerate(filters):
        trans = filters[fname]
        photometry[:, i] = simps(spectra * trans, wavelengths, axis=1) / simps(trans, wavelengths)
    return photometry

# -------------------------------------------------------------
# 5. Reconstruction framework
# -------------------------------------------------------------
def build_calibration_matrix(wavelengths, basis, filters):
    """Compute expected photometric response for unit basis spectra."""
    n_filters = len(filters)
    n_basis = basis.shape[0]
    calib = np.empty((n_filters, n_basis))
    for i, fname in enumerate(filters):
        trans = filters[fname]
        # integrate each basis spectrum over this filter
        calib[i, :] = simps(basis.T * trans, wavelengths) / simps(trans, wavelengths)
    return calib  # shape (n_filters, n_basis)

def reconstruct_spectrum(photometry, calib_matrix, wavelengths, basis):
    """Reconstruct spectra from photometry by solving for coefficients."""
    # Solve linear system: photometry = coeffs @ calib_matrix.T
    model = LinearRegression(fit_intercept=False)
    model.fit(calib_matrix.T, photometry.T)
    coeffs_hat = model.coef_.T   # shape (n_samples, n_basis)
    recon_spectra = coeffs_hat @ basis.T
    return recon_spectra, coeffs_hat

# -------------------------------------------------------------
# 6. Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.arange(400, 800, 5)  # nm

    # Basis functions
    basis = spectral_basis(wav)

    # Generate synthetic data
    n_samples = 100
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, wav)

    # Filters
    filter_names = ['U', 'B', 'V', 'R', 'I']
    filt_trans = filter_transmissions(wav, filter_names)

    # Photometry
    phot = compute_photometry(spectra, wav, filt_trans)

    # Calibration matrix
    calib_mat = build_calibration_matrix(wav, basis, filt_trans)

    # Reconstruct spectra
    recon_spec, est_coeffs = reconstruct_spectrum(phot, calib_mat, wav, basis)

    # Simple evaluation
    mse = np.mean((spectra - recon_spec)**2)
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"First 5 true vs estimated coefficients:")
    print(true_coeffs[:5])
    print(est_coeffs[:5])