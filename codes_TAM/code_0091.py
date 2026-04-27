import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ----------------------------
# 1. Spectral model definition
# ----------------------------
def gaussian_basis(wavelengths, n_basis, center_min, center_max, sigma_min, sigma_max):
    """Generate a matrix of Gaussian basis functions."""
    centers = np.linspace(center_min, center_max, n_basis)
    sigmas  = np.linspace(sigma_min, sigma_max, n_basis)
    basis = np.zeros((len(wavelengths), n_basis))
    for i, (c, s) in enumerate(zip(centers, sigmas)):
        basis[:, i] = norm.pdf(wavelengths, loc=c, scale=s)
    return basis

# ----------------------------
# 2. Synthetic spectra
# ----------------------------
def generate_synthetic_spectra(n_samples, basis, noise_std=0.01, rng=None):
    """Create synthetic spectra as random linear combinations of basis functions."""
    rng = np.random.default_rng(rng)
    coeffs = rng.standard_normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

# ----------------------------
# 3. Photometric data
# ----------------------------
def generate_filter_matrix(n_filters, wavelengths, rng=None):
    """Generate simple top-hat filter transmission curves."""
    rng = np.random.default_rng(rng)
    filt_mat = np.zeros((n_filters, len(wavelengths)))
    for i in range(n_filters):
        # Random filter center and width
        c = rng.uniform(wavelengths.min(), wavelengths.max())
        w = rng.uniform(40, 120)  # filter width
        filt = norm.pdf(wavelengths, loc=c, scale=w/6.0)  # approx top-hat
        filt_mat[i] = filt / filt.sum()
    return filt_mat

def generate_photometry(spectra, filt_mat, noise_std=0.02, rng=None):
    """Simulate photometric fluxes by integrating spectra with filter curves."""
    rng = np.random.default_rng(rng)
    phot = spectra @ filt_mat.T
    phot += rng.normal(scale=noise_std, size=phot.shape)
    return phot

# ----------------------------
# 4. Reconstruction
# ----------------------------
def reconstruct_coefficients(phot, filt_mat, basis, reg_alpha=0.1):
    """
    Reconstruct basis coefficients from photometric data using Ridge regression.
    Returns the estimated coefficients matrix.
    """
    # Effective response matrix: M = F @ B  (filters * basis)
    M = filt_mat @ basis
    # Fit Ridge regression for each sample independently
    recon_coeffs = []
    for y in phot:
        ridge = Ridge(alpha=reg_alpha, fit_intercept=False)
        ridge.fit(M, y)
        recon_coeffs.append(ridge.coef_)
    return np.vstack(recon_coeffs)

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from basis coefficients."""
    return coeffs @ basis.T

# ----------------------------
# Main script
# ----------------------------
if __name__ == "__main__":
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)

    # Wavelength grid
    wavelengths = np.linspace(400.0, 800.0, 400)

    # Basis functions
    n_basis = 5
    basis = gaussian_basis(
        wavelengths,
        n_basis=n_basis,
        center_min=450.0,
        center_max=750.0,
        sigma_min=10.0,
        sigma_max=30.0
    )

    # Generate synthetic spectra
    n_samples = 20
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, basis, noise_std=0.01, rng=rng
    )

    # Generate filter set
    n_filters = 4
    filt_mat = generate_filter_matrix(n_filters, wavelengths, rng=rng)

    # Generate photometric observations
    phot = generate_photometry(spectra_true, filt_mat, noise_std=0.02, rng=rng)

    # Reconstruction
    coeffs_recon = reconstruct_coefficients(phot, filt_mat, basis, reg_alpha=0.05)
    spectra_recon = reconstruct_spectra(coeffs_recon, basis)

    # Evaluation: mean squared error per sample
    mse = np.mean((spectra_true - spectra_recon)**2, axis=1)
    print("Mean squared errors per sample:")
    print(mse)

    # Example output for first spectrum
    idx = 0
    print("\nFirst true spectrum (first 10 points):")
    print(spectra_true[idx, :10])
    print("\nReconstructed spectrum (first 10 points):")
    print(spectra_recon[idx, :10])