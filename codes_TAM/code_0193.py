import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 1. Define a simple spectral model: linear combination of Gaussian basis functions
def build_basis(wavelengths, n_bases=5):
    """Build Gaussian basis functions over wavelengths."""
    mu = np.linspace(wavelengths[0], wavelengths[-1], n_bases)
    sigma = (wavelengths[-1] - wavelengths[0]) / (4 * n_bases)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - mu[None, :]) / sigma)**2)
    return basis

def synthesize_spectrum(basis, coeffs):
    """Generate a synthetic spectrum from basis and coefficients."""
    return basis @ coeffs

# 2. Generate synthetic spectra
def generate_synthetic_data(num_spectra=10, n_bases=5, noise_std=0.01):
    wavelengths = np.linspace(400, 700, 300)  # 400-700 nm
    basis = build_basis(wavelengths, n_bases)
    spectra = []
    true_coeffs = []
    for _ in range(num_spectra):
        coeffs = np.random.rand(n_bases)
        spec = synthesize_spectrum(basis, coeffs)
        spec += np.random.normal(scale=noise_std, size=spec.shape)
        spectra.append(spec)
        true_coeffs.append(coeffs)
    return wavelengths, np.array(spectra), np.array(true_coeffs)

# 3. Generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filters):
    """
    Filters: [n_filters, n_wavelengths]
    returns photometry (n_samples, n_filters)
    """
    trans = np.array(filters)  # shape (n_filters, n_wavelengths)
    # integrate spectra over each filter
    phot = spectra @ trans.T
    return phot

# 4. Reconstruct synthetic spectrum from photometry
def reconstruct_from_photometry(phot, wavelengths, filters, alpha=1.0):
    # Build basis again
    basis = build_basis(wavelengths, n_bases=5)
    # Ridge regression: solve for coeffs using photometric info
    # We have phot = X * coeffs + noise where X = basis * trans
? 
We have to finish code. Need to compute matrix X: For each sample, spect = basis@coeffs; phot = spect @ trans.T = (basis@coeffs) @ trans.T = coeffs @ (basis.T @ trans.T)? Let's derive: basis shape (n_wave, n_bases). trans shape (n_filters, n_wave). So basis @ coeffs -> shape (n_wave). Then phot = spect @ trans.T -> shape (n_filters). That equals (basis @ coeffs) dot trans.T. Equivalent to coeffs @ (basis.T @ trans.T). Compute M = basis.T @ trans.T => shape (n_bases, n_filters). Then phot = coeffs @ M. So we can regress: given phot (samples x n_filters) and M, find coeffs (samples x n_bases) via ridge: Solve for coeffs minimizing ||phot - coeffs @ M||^2 + alpha||coeffs||^2. This is linear regression. Use Ridge with fit_intercept=False, use .fit on M.T? Actually Ridge expects X coeff matrix shape (n_samples, n_features) and y shape (n_samples, n_targets). Here we want to predict phot from coeffs. But we can transpose: We need to solve for coeffs per sample. Use Ridge per sample? Instead use sklearn's LinearRegression for each sample? We can solve analytic closed form: coeffs = phot @ M.T @ inv(M @ M.T + alpha I). Let's implement that.

Define reconstruct_from_photometry returning reconstructed spectra: coeffs @ basis.T? Wait basis shape (n_wave, n_bases). So spectrum = basis @ coeffs. We'll output reconstructed spectra array.

Let's implement.

Also include main section generating filters: simple top-hat filters. And test.

Let's craft final code.assistantfinal code