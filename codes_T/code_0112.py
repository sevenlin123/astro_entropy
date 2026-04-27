import numpy as np
from scipy.special import erf
from sklearn.linear_model import Ridge

# --------------------------------------------
# 1. Spectral model: basis spectra
# --------------------------------------------
def generate_basis(num_basis, wavelengths, rng):
    """Generate a set of positive, smooth basis spectra."""
    basis = []
    for _ in range(num_basis):
        amp   = rng.uniform(0.5, 1.5)
        cen   = rng.uniform(wavelengths[0], wavelengths[-1])
        wid   = rng.uniform(30, 100)
        # Gaussian envelope
        spec = amp * np.exp(-0.5 * ((wavelengths - cen)/wid)**2)
        basis.append(spec)
    return np.vstack(basis)          # shape (num_basis, len(wavelengths))

# --------------------------------------------
# 2. Filter responses
# --------------------------------------------
def generate_filters(num_filters, wavelengths, rng):
    """Generate smooth, normalized filter transmission curves."""
    filters = []
    for _ in range(num_filters):
        cen   = rng.uniform(wavelengths[0], wavelengths[-1])
        wid   = rng.uniform(50, 120)
        trans = np.exp(-0.5 * ((wavelengths - cen)/wid)**2)
        trans /= trans.sum()           # normalize to unit area
        filters.append(trans)
    return np.vstack(filters)         # shape (num_filters, len(wavelengths))

# --------------------------------------------
# 3. Synthetic spectra
# --------------------------------------------
def synthesize_spectra(num_spectra, basis, rng):
    """Randomly mix basis spectra to create synthetic spectra."""
    coeffs = rng.uniform(0.0, 1.0, size=(num_spectra, basis.shape[0]))
    spectra = coeffs @ basis           # shape (num_spectra, len(wavelengths))
    return spectra, coeffs

# --------------------------------------------
# 4. Photometric fluxes
# --------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra over each filter to obtain photometric fluxes."""
    delta = wavelengths[1] - wavelengths[0]
    # pre‑compute integrand for each filter-sample pair
    fluxes = np.einsum('ij,lj->li', spectra, filters) * delta
    return fluxes                        # shape (num_spectra, num_filters)

# --------------------------------------------
# 5. Reconstruction
# --------------------------------------------
def reconstruct_spectra(photometry, basis, filters, wavelengths, alpha=0.01):
    """
    Reconstruct spectra (coefficients and fluxes) from photometric data.
    Uses a ridge regression solver for the linear inverse problem.
    """
    # Build the design matrix A : each row = integral of basis * filter
    delta = wavelengths[1] - wavelengths[0]
    A = np.einsum('ij,lj->li', basis, filters) * delta   # shape (num_filters, num_basis)
    
    coeffs_recon = np.zeros((photometry.shape[0], basis.shape[0]))
    spectra_recon = np.zeros_like(photometry)
    
    for i, flux in enumerate(photometry):
        # Solve A * coeffs = flux
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(A.T, flux)                      # A.T has shape (num_basis, num_filters)
        coeffs_recon[i] = model.coef_
        spectra_recon[i] = coeffs_recon[i] @ basis
    
    return coeffs_recon, spectra_recon

# --------------------------------------------
# 6. Demo
# --------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)
    
    # Wavelength grid (nm)
    wavelengths = np.linspace(300, 800, 500)
    
    # Generate basis spectra and filter responses
    basis = generate_basis(num_basis=5, wavelengths=wavelengths, rng=rng)
    filters = generate_filters(num_filters=3, wavelengths=wavelengths, rng=rng)
    
    # Create synthetic spectra and corresponding photometry
    spectra_true, coeffs_true = synthesize_spectra(num_spectra=10, basis=basis, rng=rng)
    photometry = compute_photometry(spectra_true, filters, wavelengths)
    
    # Reconstruct spectra from photometric data
    coeffs_rec, spectra_rec = reconstruct_spectra(
        photometry, basis, filters, wavelengths, alpha=0.01)
    
    # Simple error assessment
    recon_error = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {recon_error:.4e}")
    print("\nTrue coefficients vs. reconstructed coefficients (first star):")
    print("True :", coeffs_true[0])
    print("Rec  :", coeffs_rec[0])

if __name__ == "__main__":
    main()