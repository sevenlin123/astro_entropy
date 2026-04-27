import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------
# 1. Spectral model: Gaussian basis functions
# ----------------------------------------------------------------------
def build_basis(wavelengths, n_basis=5, rng=None):
    """Return matrix X of shape (len(wavelengths), n_basis) where each column
    is a Gaussian basis function evaluated at the wavelengths."""
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(400, 800, size=n_basis)
    widths  = rng.uniform(10, 50,  size=n_basis)
    X = np.zeros((len(wavelengths), n_basis))
    for i, (c, w) in enumerate(zip(centers, widths)):
        X[:, i] = norm.pdf(wavelengths, loc=c, scale=w)
    return X

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_spectra(n_samples, X, rng=None):
    """Generate n_samples synthetic spectra as linear combinations of basis X."""
    if rng is None:
        rng = np.random.default_rng()
    # Random coefficients ~ N(0,1)
    coeffs = rng.standard_normal(size=(n_samples, X.shape[1]))
    spectra = coeffs @ X.T   # shape (n_samples, n_wavelengths)
    return spectra

# ----------------------------------------------------------------------
# 3. Photometric filters
# ----------------------------------------------------------------------
def build_filters(wavelengths, filter_centers, filter_width=20.0):
    """Return filter matrix F of shape (n_filters, len(wavelengths))."""
    F = np.array([norm.pdf(wavelengths, loc=fc, scale=filter_width)
                  for fc in filter_centers])
    return F

def photometry_from_spectra(spectra, F):
    """Compute synthetic photometric fluxes: P = F @ spectra.T"""
    return spectra @ F.T   # shape (n_samples, n_filters)

# ----------------------------------------------------------------------
# 4. Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra(photon_data, X, F):
    """Reconstruct spectra from photometric data using linear least‑squares."""
    # Pre‑compute matrix relating basis coefficients to photometry
    FX = F @ X.T            # shape (n_filters, n_basis)
    n_samples = photon_data.shape[0]
    reconstructed = np.empty_like(photon_data, shape=(n_samples, X.shape[0]))
    for i in range(n_samples):
        c, *_ = np.linalg.lstsq(FX, photon_data[i], rcond=None)
        reconstructed[i] = X @ c
    return reconstructed

# ----------------------------------------------------------------------
# 5. Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid: 400–800 nm in 5‑nm steps
    wavelengths = np.arange(400, 801, 5)

    # Build basis
    X = build_basis(wavelengths, n_basis=5, rng=rng)

    # Generate synthetic spectra
    n_samples = 50
    spectra_true = generate_spectra(n_samples, X, rng=rng)

    # Define photometric filters (B, V, R)
    filter_centers = np.array([440, 550, 660])  # nm
    F = build_filters(wavelengths, filter_centers, filter_width=20.0)

    # Generate photometric data
    photometry = photometry_from_spectra(spectra_true, F)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(photometry, X, F)

    # Evaluate reconstruction
    mae = np.mean(np.abs(spectra_true - spectra_rec))
    print(f"Mean absolute error per wavelength: {mae:.3f}")