import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1. Define a simple spectral model: linear combination of Gaussian basis functions
def gaussian_basis(wavelengths, centers, widths):
    """Generate Gaussian basis matrix."""
    return np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :]) ** 2)

def spectral_model(basis, coefficients):
    """Linear combination of basis functions."""
    return basis @ coefficients

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths, centers, widths, noise_level=0.01):
    basis = gaussian_basis(wavelengths, centers, widths)
    coeffs = np.random.randn(n_samples, len(centers))
    spectra = basis @ coeffs.T
    spectra += np.random.normal(scale=noise_level, size=spectra.shape)
    return spectra.T, coeffs  # return spectra shape (n_wavelengths, n_samples)

# 3. Generate photometric data from synthetic spectra
def photometric_flux(spectra, filter_transmission, wavelengths):
    """Integrate flux through filter."""
    filt_int = np.trapz(filter_transmission, wavelengths)
    return np.trapz(spectra * filter_transmission[None, :], wavelengths) / filt_int

def compute_photometry(spectra, filters, wavelengths):
    fluxes = []
    for filt in filters:
        flux = photometric_flux(spectra, filt, wavelengths)
        fluxes.append(flux)
    return np.vstack(fluxes).T  # (n_samples, n_filters)

# 4. reconstruct spectrum from photometry using Lasso regression
def reconstruct_from_photometry(photometry, basis, wavelengths, n_iter=1000):
    # Standardize photometry to mean=.. etc
    scaler = StandardScaler()
    y_std = scaler.fit_transform(photometry)
    # 0-10?  #?? maybe 0? 
    **??**?

We have to output final runnable code. Let's fix issues. Use Lasso to predict coefficients from photometry. Need basis matrix? We can regress each wavelength against photometry via lasso: For each wavelength, fit Lasso model using photometry as features to predict flux at that wavelength across samples. Then reconstruct spectrum by predicting flux per sample across wavelengths. But we can also try to recover coefficients via linear regression with regularization. For simplicity, reconstruct spectrum by regressing photometry to coefficients using lasso: For each coefficient dimension, fit lasso with photometry as X, coeffs as y. Then for new photometry, predict coefficients then compute spectrum via basis. Since synthetic data we know coeffs. We'll implement reconstruction for all samples.

Let's craft functions:

- reconstruct_coefficients(photometry, coeffs) maybe using ridge.

But we just need a single function that takes photometry, basis, wavelengths returns reconstructed spectra. We'll generate training coeffs via synthetic data and fit lasso models per coefficient dimension.

Simplify: Use sklearn's LassoCV but we skip cross-val. Use Lasso(alpha=0.1). Fit for each coefficient. Then predict coefficients for each sample. Then compute spectrum. That's okay.

Also we need filter transmissions: create random filter bandpasses. Provide few filters.

Let's write full script. Ensure import of Lasso. We'll also need to set random seed for reproducibility.

Let's craft.