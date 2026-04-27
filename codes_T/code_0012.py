import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# --- 1. Spectral model -------------------------------------------------------
def create_wavelength_grid(n_points=1000, wl_min=350, wl_max=1050):
    """Create a wavelength grid (nm)."""
    return np.linspace(wl_min, wl_max, n_points)

def gaussian_basis(wl, center, width):
    """Gaussian basis function."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_basis_set(wl, n_bases=10):
    """Build a set of Gaussian basis functions."""
    centers = np.linspace(wl.min(), wl.max(), n_bases)
    width = (wl.max() - wl.min()) / (n_bases * 4)
    basis = [gaussian_basis(wl, c, width) for c in centers]
    return np.array(basis)   # shape (n_bases, len(wl))

# --- 2. Generate synthetic spectra -----------------------------------------
def generate_random_coeffs(n_samples, n_bases, rng=np.random.default_rng(42)):
    """Random coefficients for spectra."""
    return rng.normal(size=(n_samples, n_bases)) * 0.5 + 1.0

def synthesize_spectra(basis, coeffs):
    """Linear combination of basis functions."""
    return coeffs @ basis  # shape (n_samples, len(wl))

# --- 3. Photometric data ----------------------------------------------------
def gaussian_filter(wl, center, width, amplitude=1.0):
    """Gaussian filter transmission curve."""
    return amplitude * np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_filters(wl, n_filters=5):
    """Build simple filter set."""
    centers = np.linspace(wl.min()+50, wl.max()-50, n_filters)
    width = (wl.max() - wl.min()) / (n_filters * 6)
    filters = [gaussian_filter(wl, c, width) for c in centers]
    return np.array(filters)  # shape (n_filters, len(wl))

def integrate_flux(spectrum, filt, wl):
    """Integrate flux of spectrum through filter."""
    return simps(spectrum * filt, wl)

def compute_photometry(spectra, filters, wl):
    """Compute photometric fluxes for each spectrum."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        phot[:, i] = simps(spectra * filters[i], wl, axis=1)
    return phot

# --- 4. Reconstruction ------------------------------------------------------
def reconstruct_from_photometry(photometry, filters, wl):
    """
    Reconstruct spectra by solving a linear system:
    photometry = (basis_matrix @ filters.T).coeffs
    We approximate inverse mapping with linear regression.
    """
    # Build matrix mapping coefficients to photometry
    basis = build_basis_set(wl)
    basis_to_phot = (basis @ filters.T).T  # shape (n_filters, n_bases)
    
    # Fit linear regression from photometry to coefficients
    reg = LinearRegression(fit_intercept=False)
    reg.fit(basis_to_phot, np.eye(basis_to_phot.shape[1]))
    
    # Predict coefficients from observed photometry
    coeff_pred = reg.predict(photometry)  # shape (n_samples, n_bases)
    # Reconstruct spectra
    reconstructed = coeff_pred @ basis
    return reconstructed

# --- 5. Main routine --------------------------------------------------------
def main():
    rng = np.random.default_rng(12345)
    wl = create_wavelength_grid()
    basis = build_basis_set(wl)
    
    n_samples = 200
    n_bases = basis.shape[0]
    coeffs_true = generate_random_coeffs(n_samples, n_bases, rng)
    spectra_true = synthesize_spectra(basis, coeffs_true)
    
    filters = build_filters(wl)
    phot = compute_photometry(spectra_true, filters, wl)
    
    spectra_recon = reconstruct_from_photometry(phot, filters, wl)
    
    # Evaluate reconstruction
    mse = np.mean((spectra_true - spectra_recon) ** 2)
    print(f"Reconstruction MSE: {mse:.4e}")
    
    # Show a few examples
    idx = [0, 1, 2]
    for i in idx:
        diff = spectra_true[i] - spectra_recon[i]
        print(f"Spectrum {i}: max abs diff = {np.max(np.abs(diff)):.4f}")

if __name__ == "__main__":
    main()