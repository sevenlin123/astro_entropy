import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# -----------------------------
# Spectral model definition
# -----------------------------

def gaussian_basis(wave, centers, widths):
    """Generate Gaussian basis functions."""
    G = np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / widths[None, :]) ** 2)
    return G

# -----------------------------
# Synthetic data generation
# -----------------------------

def generate_synthetic_spectra(n_samples, basis_funcs):
    """Random linear combination of basis functions."""
    coeffs = np.random.randn(n_samples, basis_funcs.shape[1])
    spectra = coeffs @ basis_funcs.T
    return spectra, coeffs

def gaussian_filter(wave, center, width):
    """Simple Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wave - center) / width) ** 2)

def define_filters(wave, filter_centers, filter_widths):
    """Create filter transmission curves."""
    return [gaussian_filter(wave, c, w) for c, w in zip(filter_centers, filter_widths)]

def photometric_flux(spectrum, filt, wave):
    """Integrate spectrum over filter transmission."""
    num = simps(spectrum * filt, wave)
    denom = simps(filt, wave)
    return num / denom

def compute_photometry(spectra, filters, wave):
    """Compute photometric fluxes for all spectra."""
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    fluxes = np.empty((n_samples, n_filters))
    for i, filt in enumerate(filters):
        fluxes[:, i] = np.array([photometric_flux(spec, filt, wave) for spec in spectra])
    return fluxes

# -----------------------------
# Reconstruction framework
# -----------------------------

def build_filter_matrix(basis_funcs, filters, wave):
    """Project basis functions onto filters."""
    n_basis = basis_funcs.shape[1]
    n_filters = len(filters)
    F = np.empty((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            F[i, j] = photometric_flux(basis_funcs[:, j], filt, wave)
    return F

def reconstruct_spectra(photometry, filter_matrix, basis_funcs, alpha=1.0):
    """Recover spectrum coefficients via ridge regression."""
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(filter_matrix, photometry.T)
    coeffs_recon = model.coef_.T
    spectra_recon = coeffs_recon @ basis_funcs.T
    return spectra_recon, coeffs_recon

# -----------------------------
# Main demonstration
# -----------------------------

def main():
    # Wavelength grid
    wave = np.linspace(300, 800, 500)  # nm
    
    # Basis functions
    n_basis = 5
    centers = np.linspace(350, 750, n_basis)
    widths = np.full(n_basis, 50.0)
    basis_funcs = gaussian_basis(wave, centers, widths)
    
    # Generate synthetic spectra
    n_samples = 100
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis_funcs)
    
    # Define filters (UBVRI-like)
    filter_centers = [360, 440, 550, 660, 780]
    filter_widths = [40, 40, 40, 40, 40]
    filters = define_filters(wave, filter_centers, filter_widths)
    
    # Compute photometry
    photometry = compute_photometry(spectra_true, filters, wave)
    
    # Build filter matrix
    filter_matrix = build_filter_matrix(basis_funcs, filters, wave)
    
    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filter_matrix, basis_funcs, alpha=0.1)
    
    # Simple evaluation
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error between true and reconstructed spectra: {mse:.4f}")

if __name__ == "__main__":
    main()