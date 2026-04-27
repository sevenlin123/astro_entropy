import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Define a spectral model – here a set of Gaussian basis functions
# ------------------------------------------------------------------
def create_basis(wavelengths, n_basis):
    """Create `n_basis` Gaussian basis functions over the given wavelengths."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (2 * n_basis)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths) ** 2)
        basis.append(g)
    return np.array(basis)          # shape (n_basis, len(wavelengths))

# ------------------------------------------------------------------
# 2. Generate synthetic spectra as random combinations of the basis
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, basis):
    """Generate `n_samples` synthetic spectra using random linear combos."""
    coeffs = np.random.randn(n_samples, basis.shape[0])  # random coefficients
    spectra = coeffs @ basis                       # matrix multiplication
    return spectra, coeffs

# ------------------------------------------------------------------
# 3. Define photometric filters and compute photometry from spectra
# ------------------------------------------------------------------
def create_filters(n_filters, wavelengths):
    """Create `n_filters` simple Gaussian filter curves."""
    centers = np.linspace(wavelengths.min() + 0.1*(wavelengths.max()-wavelengths.min()),
                          wavelengths.max() - 0.1*(wavelengths.max()-wavelengths.min()),
                          n_filters)
    widths = (wavelengths.max() - wavelengths.min()) / (10 * n_filters)
    filters = []
    for c in centers:
        filt = np.exp(-0.5 * ((wavelengths - c) / widths) ** 2)
        filt /= filt.sum()               # normalise
        filters.append(filt)
    return np.array(filters)             # shape (n_filters, len(wavelengths))

def compute_photometry(spectra, filters):
    """Integrate each spectrum through each filter to obtain photometric fluxes."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        phot[:, i] = np.array([simps(s * filters[i], x=np.arange(len(s))) for s in spectra])
    return phot

# ------------------------------------------------------------------
# 4. Reconstruct spectra from photometric measurements
# ------------------------------------------------------------------
def reconstruct_spectra(photometry, filters, basis, alpha=1.0):
    """
    Train a Ridge regression model to predict spectral coefficients
    from photometric measurements, then reconstruct spectra.
    """
    # Estimate coefficients: we learn a mapping phot -> coeffs
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(photometry, np.linalg.pinv(basis.T) @ np.random.randn(basis.shape[0], basis.shape[0]))
    # Predict coefficients for new photometry
    coeff_pred = reg.predict(photometry)
    # Reconstruct spectra
    spectra_rec = coeff_pred @ basis
    return spectra_rec, coeff_pred

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (e.g., 400–800 nm sampled at 1 nm)
    wav = np.arange(400, 801, 1)

    # Build basis
    n_basis = 20
    basis = create_basis(wav, n_basis)

    # Generate synthetic spectra
    n_samples = 200
    spectra, coeffs_true = generate_synthetic_spectra(n_samples, wav, basis)

    # Create filters
    n_filters = 5
    filters = create_filters(n_filters, wav)

    # Compute photometry
    phot = compute_photometry(spectra, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_est = reconstruct_spectra(phot, filters, basis, alpha=1.0)

    # Evaluate (simple mean squared error)
    mse = np.mean((spectra - spectra_rec)**2)
    print(f"Reconstruction MSE: {mse:.4f}")