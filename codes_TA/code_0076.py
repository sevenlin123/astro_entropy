import numpy as np
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Define wavelength grid
# ------------------------------------------------------------
wavelengths = np.arange(400, 801, 0.5)  # 400–800 nm, 0.5 nm step
n_wave = len(wavelengths)

# ------------------------------------------------------------
# 2. Define synthetic spectral model
# ------------------------------------------------------------
def synthetic_spectrum():
    """Generate a single synthetic spectrum."""
    # Continuum
    cont = 1.0 + 0.05 * np.sin((wavelengths - 400) / 100.0)
    # Random Gaussian absorption/emission features
    n_features = np.random.randint(3, 7)
    spec = cont.copy()
    for _ in range(n_features):
        cen = np.random.uniform(420, 780)
        amp = np.random.uniform(0.1, 0.3)
        sigma = np.random.uniform(5, 15)
        spec += amp * np.exp(-0.5 * ((wavelengths - cen) / sigma) ** 2)
    # Add small Gaussian noise
    noise = np.random.normal(scale=0.01, size=n_wave)
    return spec + noise

def generate_synthetic_spectra(n_samples=50):
    """Generate multiple synthetic spectra."""
    return np.array([synthetic_spectrum() for _ in range(n_samples)])

# ------------------------------------------------------------
# 3. Define photometric filters
# ------------------------------------------------------------
def top_hat(center, width):
    """Return a top‑hat transmission curve."""
    return np.where(np.abs(wavelengths - center) <= width / 2.0, 1.0, 0.0)

filters = {
    'F500': top_hat(500, 50),
    'F600': top_hat(600, 50),
    'F700': top_hat(700, 50),
    'F800': top_hat(800, 50),
}

filter_names = list(filters.keys())
n_filters = len(filter_names)

# ------------------------------------------------------------
# 4. Compute photometric fluxes from spectra
# ------------------------------------------------------------
def photometry_from_spectra(spectra):
    """
    Integrate each spectrum through all filters.
    Returns array of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    phot = np.empty((n_samples, n_filters))
    for i, name in enumerate(filter_names):
        trans = filters[name]
        # Simple trapezoidal integration
        phot[:, i] = np.trapz(spectra * trans, wavelengths, axis=1)
    return phot

# ------------------------------------------------------------
# 5. Reconstruction framework
# ------------------------------------------------------------
def reconstruct_spectra(photometry, n_basis=10, reg_alpha=1e-3):
    """
    Reconstruct spectra from photometric fluxes using polynomial basis.
    
    Parameters
    ----------
    photometry : array of shape (n_samples, n_filters)
    n_basis   : number of polynomial basis functions (including constant term)
    reg_alpha : ridge regularisation parameter
    
    Returns
    -------
    recon_spec : array of shape (n_samples, n_wave)
    """
    # Construct polynomial basis functions (monomials up to degree n_basis-1)
    X_basis = np.vstack([wavelengths**i for i in range(n_basis)])  # shape (n_basis, n_wave)
    
    # Compute design matrix A (n_filters, n_basis)
    A = np.zeros((n_filters, n_basis))
    for i, name in enumerate(filter_names):
        trans = filters[name]
        # Integral of basis * transmission over wavelength
        for j in range(n_basis):
            A[i, j] = np.trapz(X_basis[j] * trans, wavelengths)
    
    # Pre‑compute regularised inverse
    M = A.T @ A + reg_alpha * np.eye(n_basis)
    invM = np.linalg.inv(M)
    ATy = A.T  # will multiply by photometry later
    
    # Solve for coefficients for all samples
    coeffs = (invM @ ATy) @ photometry.T  # shape (n_basis, n_samples)
    
    # Reconstruct spectra
    recon_spec = X_basis.T @ coeffs  # shape (n_wave, n_samples)
    return recon_spec.T

# ------------------------------------------------------------
# 6. Main routine
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data
    spectra = generate_synthetic_spectra(n_samples=50)
    phot = photometry_from_spectra(spectra)
    
    # Reconstruct spectra from photometry
    recon = reconstruct_spectra(phot, n_basis=10, reg_alpha=1e-3)
    
    # Evaluate reconstruction quality
    mse = np.mean((spectra - recon) ** 2)
    print(f"Mean squared error of reconstruction: {mse:.6f}")