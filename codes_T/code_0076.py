import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------
# Basis and filter definitions
# ----------------------------------------------------------
def generate_basis(num_wavelengths, num_basis):
    """Generate orthonormal basis spectra."""
    wavelengths = np.linspace(400, 1000, num_wavelengths)  # nm
    basis = np.array([np.exp(-((wavelengths - w0)/30)**2)
                      for w0 in np.linspace(450, 950, num_basis)])
    # Orthogonalize and normalize
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T
    return wavelengths, basis

def generate_filters(num_filters, num_wavelengths):
    """Generate simple Gaussian transmission curves."""
    wavelengths = np.linspace(400, 1000, num_wavelengths)
    filters = []
    for i in range(num_filters):
        center = 400 + i * (600/(num_filters-1))
        filt = np.exp(-((wavelengths - center)/50)**2)
        filt /= np.max(filt)
        filters.append(filt)
    return np.array(filters)  # shape (num_filters, num_wavelengths)

# ----------------------------------------------------------
# Synthetic spectrum generation
# ----------------------------------------------------------
def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return coeffs @ basis

# ----------------------------------------------------------
# Photometric integration
# ----------------------------------------------------------
def compute_photometry(spectrum, filters):
    """Integrate spectrum against each filter curve."""
    phot = []
    for filt in filters:
        flux = simps(spectrum * filt, axis=-1)
        phot.append(flux)
    return np.array(phot)

# ----------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------
def reconstruct_coeffs(photometry, filters, basis, alpha=1e-3):
    """
    Solve for coefficients that best reproduce the photometry.
    Uses Ridge regression (regularized least squares).
    """
    # Build design matrix: integral of basis * filter
    design = np.array([[simps(basis[j]*filters[i], axis=-1) 
                        for j in range(basis.shape[0])]
                       for i in range(filters.shape[0])])
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design, photometry)
    return ridge.coef_

# ----------------------------------------------------------
# Demo
# ----------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    num_wl = 200            # wavelength samples
    num_basis = 5           # basis spectra
    num_filters = 4         # photometric bands
    num_spectra = 10        # synthetic sources

    # Generate basis and filters
    wavelengths, basis = generate_basis(num_wl, num_basis)
    filters = generate_filters(num_filters, num_wl)

    # Create true coefficients and spectra
    true_coeffs = np.random.randn(num_spectra, num_basis)
    spectra = np.array([synthesize_spectrum(basis, c) for c in true_coeffs])

    # Compute photometry
    photometry = np.array([compute_photometry(spec, filters) for spec in spectra])

    # Reconstruct coefficients
    recon_coeffs = np.array([reconstruct_coeffs(p, filters, basis) for p in photometry])

    # Evaluate reconstruction error
    recon_spectra = np.array([synthesize_spectrum(basis, c) for c in recon_coeffs])
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.3e}")