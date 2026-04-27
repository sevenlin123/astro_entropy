import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model – basis functions
# ----------------------------------------------------------------------
def create_gaussian_basis(n_basis, wl):
    """Create a set of gaussian basis functions over wavelength grid."""
    np.random.seed(0)
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    widths  = np.linspace((wl.max()-wl.min())/8,
                          (wl.max()-wl.min())/4,
                          n_basis)
    basis = np.zeros((n_basis, len(wl)))
    for k in range(n_basis):
        basis[k] = np.exp(-0.5 * ((wl - centers[k]) / widths[k])**2)
    return basis

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_spectra(n_spec, basis, coeff_sigma=0.5, noise_sigma=0.01):
    """Generate spectra as linear combos of basis functions plus noise."""
    np.random.seed(1)
    coeffs = np.random.randn(n_spec, basis.shape[0]) * coeff_sigma
    spectra = coeffs @ basis
    spectra += np.random.randn(*spectra.shape) * noise_sigma
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Create photometric filters (simple top‑hat)
# ----------------------------------------------------------------------
def create_tophat_filters(n_filter, wl):
    """Create n_filter top‑hat filters over wavelength grid."""
    np.random.seed(2)
    bands = np.linspace(wl.min(), wl.max(), n_filter+1)
    filters = np.zeros((n_filter, len(wl)))
    for i in range(n_filter):
        mask = (wl >= bands[i]) & (wl < bands[i+1])
        filters[i, mask] = 1.0
    # Normalise to unit area
    filters /= filters.sum(axis=1)[:, None]
    return filters

# ----------------------------------------------------------------------
# 4. Compute photometry from spectra
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """Integrate spectra through filters to obtain fluxes."""
    # flux = ∫ S(λ) R(λ) dλ / ∫ R(λ)dλ  (filters already normalised)
    return spectra @ filters.T

# ----------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_from_photometry(phot, filters, basis):
    """
    Reconstruct spectra via linear regression:
        phot = M @ coeffs      where M_ij = ∫ basis_j * filter_i dλ
    """
    # Build design matrix M
    M = np.zeros((filters.shape[0], basis.shape[0]))
    for i in range(filters.shape[0]):
        for j in range(basis.shape[0]):
            M[i, j] = np.trapz(basis[j] * filters[i], axis=0)

    # Linear regression without intercept
    lr = LinearRegression(fit_intercept=False)
    lr.fit(M, phot)
    coeff_est = lr.predict(M)

    # Reconstructed spectra
    recon = coeff_est @ basis
    return recon, coeff_est

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 800, 1000)  # nm

    # Build model
    n_basis   = 5
    n_specs   = 20
    n_filters = 4

    basis   = create_gaussian_basis(n_basis, wl)
    spectra, true_coeffs = generate_spectra(n_specs, basis)
    filters = create_tophat_filters(n_filters, wl)
    phot    = compute_photometry(spectra, filters)

    recon_spectra, est_coeffs = reconstruct_from_photometry(phot, filters, basis)

    # Print results
    print("True coefficients shape:", true_coeffs.shape)
    print("Estimated coefficients shape:", est_coeffs.shape)
    print("\nFirst spectrum (true vs reconstructed):")
    print(spectra[0][:10])
    print(recon_spectra[0][:10])

    # Compute reconstruction error
    err = np.mean((spectra - recon_spectra)**2)
    print(f"\nMean squared reconstruction error: {err:.6f}")