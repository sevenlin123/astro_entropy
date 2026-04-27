import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# 1. Define a simple linear spectral model: flux = sum_i w_i * basis_i(λ)
def define_basis(wavelength, n_basis=5):
    """Generate random basis functions (Gaussian peaks)."""
    np.random.seed(0)
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    widths = np.full(n_basis, (wavelength.max()-wavelength.min())/(4*n_basis))
    basis = []
    for c, s in zip(centers, widths):
        basis.append(np.exp(-0.5*((wavelength-c)/s)**2))
    return np.vstack(basis)   # shape (n_basis, n_points)

def spectral_model(coeffs, basis):
    """Given coefficients and basis matrix, compute flux."""
    return coeffs @ basis  # (n_basis,) @ (n_basis,n_points) -> (n_points,)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples=50, wavelength=np.arange(400, 701, 1)):
    """Simulate spectra with random coefficients."""
    basis = define_basis(wavelength, n_basis=5)
    true_coeffs = np.random.randn(n_samples, 5)
    spectra = np.array([spectral_model(c, basis) for c in 
                        true_coeffs])
    return wavelength, spectra, true_coeffs, basis

# 3. photometric data: create bandpasses (beyond),  &   .. 
def create_bandpasses(wavelength, number=10):
    """
        ... …
    """OK?

We have partial code; need to finish. Provide full runnable code. Requirements: define spectral model, generate synthetic spectra, generate photometric data from synthetic spectra, reconstruct synthetic spectrum from photometric data. Use numpy, scipy, scikit-learn. Clear functions.

Implement: define_basis, spectral_model, generate_synthetic_spectra. Photometry: define bandpasses as top-hat filters. Compute integrated fluxes via weighted average of spectrum. Then reconstruction: given photometry, solve least squares using basis to predict spectrum. Use Ridge regression? We can do ridge on bandpass integrals to predict coefficients. Approach: for each sample, we have photometric vector p (size m). Use training data of (p, coeffs). Fit Ridge regression to map p -> coeffs. Then predict coeffs for new photometry. Use predicted coeffs to reconstruct spectrum.

Generate synthetic photometry from spectra. Build training set same as generating spectra. Then use regression.

Let's write final code. Ensure reproducibility. Add main block. Let's produce code.assistantfinal code