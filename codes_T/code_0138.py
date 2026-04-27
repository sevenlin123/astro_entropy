import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# Define a simple spectral model: linear combination of basis spectra
def build_basis(wavelengths, n_basis=5):
    """Generate a set of synthetic basis spectra over the given wavelengths."""
    rng = np.random.default_rng(42)
    # Randomly generate base shapes
    basis = rng.standard_normal((len(wavelengths), n_basis))
    # Ensure positivity and smoothness via Gaussian smoothing
    for i in range(n_basis):
        kernel = np.exp(-0.5 * ((np.arange(len(wavelengths)) - len(wavelengths)//2) /
                                (0.1*len(wavelengths)))**2)
        basis[:, i] = np.convolve(basis[:, i], kernel, mode='same')
    return basis

def synthesize_spectrum(basis, coeffs):
    """Construct a synthetic spectrum from basis coefficients."""
    return basis @ coeffs

def generate_synthetic_data(n_samples=100, wave_min=350, wave_max=950, n_points=500):
    """generate synthetic spectra and photometric data."""
    wavelengths = np.linspace(wave_min, wave_max, n_points)
    basis = build_basis(wavelengths)
    rng = np.random.default_rng(1234)
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    # Ensure non‑negative coefficients for physical spectra
    coeffs = np.abs(coeffs)
    spectra = np.array([synthesize_spectrum(basis, c) for c in coeffs])
    # Photometric filters: narrowband at 400, 550, 700 nm
    filter_centers = np.array([400, 550, 700])
    filter_width = 30  # nm
    def apply_filter(spectrum):
        filt = np.exp(-0.5*((wavelengths-filter_centers[:,None])/filter_width)**2)
        return (spectrum * filt).sum(axis=1)
    phot = np.array([apply_filter(s) for s in spectra])
    return wavelengths, spectra, phot, basis, coeffs

def reconstruct_spectrum_from_photometry(phot, basis, wavelengths, filter_centers,
                                         filter_width=30, alpha=1e-3):
    """Use ridge regression to match photometric fluxes to basis coefficients."""
    n_basis = basis.shape[1]
    # Build design matrix: compute expected photometric fluxes for each basis
    n_filters = len(filter_centra
l)  # compute filtered flux per basis spectrum
    filter_matrix = np.zeros((n_filters, n_basis))
    for f_idx, fc in enumerate(filter_centers):
        filt = np.exp(-(wavelengths-fc)**2/(2*filter_width**2))
        filter_matrix[f_idx] = (basis * filt[:, None]).sum(axis=0)
    # ridge regression: solve for coefficients that match phot flux
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(filter_matrix.T, phot.T)
    coeffs_hat = ridge.coef_.T
    return coeffs_hat

def main():
    wavelengths, spectra, phot, basis, true_coeffs = generate_synthetic_data()
    # Recover coefficients from photometry
    filter_centers = np.array([400, 550, 700])
    coeffs_hat = reconstruct_spectrum_from_photometry(
        phot, basis, wavelengths, filter_centers)
    # Reconstruct the spectra using recovered coefficients
    spectra_recon = np.array([synthesize_spectrum(basis, c) for c in coeffs_hat])
    print("True vs reconstructed coefficient error (RMSE):",
          np.sqrt(((true_coeffs - coeffs_hat)**2).mean()))
    # Compare a sample spectrum
    idx = 0
    import matplotlib.pyplot (missing? need to skip plotting per constraints) 
We need to output final python code, no explanations, only code. Must handle syntax errors. The script must be runnable. We used matplotlib but constraint says no plotting. So remove plotting lines. Also fix syntax errors like "len(filter_centra l)" etc.

Let's write code accordingly. Use numpy, scipy, sklearn. Let's produce minimal workable script. We will keep some prints.

Let's rewrite.

Also note: we used `apply_filter` returning array shape mismatched; we need to compute integrated flux per filter; we might compute dot product with filter function. We'll produce a vector of flux per filter.

We'll implement: