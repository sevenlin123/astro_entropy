import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------- Spectral Model -----------------------

def create_basis(wavelengths, n_basis):
    """Generate simple sinusoidal basis spectra."""
    np.random.seed(0)
    basis = []
    for i in range(n_basis):
        freq = np.random.uniform(0.01, 0.05)
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.5, 1.5)
        spectrum = amp * np.sin(2*np.pi*freq*wavelengths + phase)
        # Normalize
        spectrum -= spectrum.mean()
        spectrum /= np.std(spectrum)
        basis.append(spectrum)
    return np.vstack(basis)  # shape (n_basis, n_wave)

# ----------------------- Synthetic Data -----------------------

def generate_synthetic_spectra(n_objects, basis, noise_std=0.02):
    """Generate spectra as linear combinations of basis spectra."""
    n_basis, n_wave = basis.shape
    coeffs = np.random.randn(n_objects, n_basis)
    spectra = coeffs @ basis  # shape (n_objects, n_wave)
    spectra += noise_std * np.random.randn(*spectra.shape)
    return spectra, coeffs

# ----------------------- Filters -----------------------

def create_filters(n_filters, wavelengths):
    """Create simple top‑hat filters."""
    n_wave = len(wavelengths)
    filters = np.zeros((n_filters, n_wave))
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    width = (wavelengths[-1] - wavelengths[0]) / (2*n_filters)
    for i, c in enumerate(centers):
        mask = (wavelengths >= c - width) & (wavelengths <= c + width)
        filters[i, mask] = 1.0
    return filters

# ----------------------- Photometry -----------------------

def compute_photometry(spectra, filters):
    """Integrate spectra over filters to produce photometric fluxes."""
    # Using Simpson's rule for integration
    return np.array([simps(spectra[:, i]*filters[j], x=wavelengths) 
                     for j in range(filters.shape[0])]).T

# ----------------------- Reconstruction -----------------------

def reconstruct_spectrum(photometry, filters, basis, reg=1e-2):
    """
    Reconstruct spectra from photometry by solving for basis coefficients.
    Uses Ridge regression (regularized least squares).
    """
    # Build design matrix: each column = integral of basis_k * filter_j
    n_filters, n_wave = filters.shape
    n_basis = basis.shape[0]
    design = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for k in range(n_basis):
            design[j, k] = simps(basis[k] * filters[j], x=wavelengths)
    # Solve for coefficients
    ridge = Ridge(alpha=reg, fit_intercept=False, solver='auto')
    ridge.fit(design.T, photometry.T)
    coeffs_recon = ridge.coef_.T  # shape (n_objects, n_basis)
    # Reconstruct spectra
    spectra_recon = coeffs_recon @ basis  # shape (n_objects, n_wave)
    return spectra_recon, coeffs_recon

# ----------------------- Main -----------------------

if __name__ == "__main__":
    # Define wavelength grid (nm)
    wavelengths = np.linspace(400, 1000, 1000)

    # Basis spectra
    n_basis = 5
    basis = create_basis(wavelengths, n_basis)

    # Synthetic spectra
    n_objects = 10
    spectra_true, coeffs_true = generate_synthetic_spectra(n_objects, basis)

    # Filters
    n_filters = 7
    filters = create_filters(n_filters, wavelengths)

    # Photometry
    photometry = compute_photometry(spectra_true, filters)

    # Reconstruction
    spectra_rec, coeffs_rec = reconstruct_spectrum(photometry, filters, basis)

    # Simple evaluation
    rmse = np.sqrt(((spectra_true - spectra_rec)**2).mean())
    print(f"Reconstruction RMSE: {rmse:.4f}")