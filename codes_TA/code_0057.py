import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# --------------------------------------
# Spectral model utilities
# --------------------------------------
def gaussian_basis(n_basis, wavelengths):
    """Generate smooth Gaussian basis spectra."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (2 * n_basis)
    basis = []
    for c in centers:
        spec = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(spec)
    return np.vstack(basis)  # shape (n_basis, n_wavelengths)

def synthetic_spectra(n_samples, basis, rng=None):
    """Create synthetic spectra as linear combinations of basis."""
    rng = np.random.default_rng(rng)
    coeffs = rng.uniform(0.5, 1.5, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

def create_filters(n_filters, wavelengths, rng=None):
    """Generate random Gaussian filter responses."""
    rng = np.random.default_rng(rng)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform((wavelengths[-1]-wavelengths[0])/10,
                            (wavelengths[-1]-wavelengths[0])/3)
        filt = np.exp(-0.5 * ((wavelengths - center) / width)**2)
        filters.append(filt)
    return np.vstack(filters)  # shape (n_filters, n_wavelengths)

# --------------------------------------
# Photometry computation
# --------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate spectra over filter responses.
    Returns fluxes shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    fluxes = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        filt = filters[i]
        fluxes[:, i] = simps(spectra * filt, wavelengths, axis=1)
    return fluxes

# --------------------------------------
# Spectrum reconstruction
# --------------------------------------
def build_design_matrix(filters, basis, wavelengths):
    """
    Build matrix A where A_ij = integral(filter_i * basis_j).
    """
    n_filters, n_wavelengths = filters.shape
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            A[i, j] = simps(filters[i] * basis[j], wavelengths)
    return A

def reconstruct_coefficients(photon_fluxes, design_matrix):
    """
    Solve for coefficients using least squares (non-negative).
    """
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(design_matrix.T, photon_fluxes.T)
    coeffs = reg.coef_.T  # shape (n_samples, n_basis)
    return coeffs

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from coefficients and basis."""
    return coeffs @ basis

# --------------------------------------
# Main execution
# --------------------------------------
def main():
    rng = 42
    # Wavelength grid
    wavelengths = np.linspace(400, 800, 200)  # nm

    # Spectral basis
    n_basis = 5
    basis = gaussian_basis(n_basis, wavelengths)

    # Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = synthetic_spectra(n_samples, basis, rng=rng)

    # Create photometric filters
    n_filters = 3
    filters = create_filters(n_filters, wavelengths, rng=rng)

    # Compute photometric fluxes
    fluxes = compute_photometry(spectra, filters, wavelengths)

    # Build design matrix and reconstruct coefficients
    design_matrix = build_design_matrix(filters, basis, wavelengths)
    recon_coeffs = reconstruct_coefficients(fluxes, design_matrix)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(recon_coeffs, basis)

    # Evaluate reconstruction error
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.4e}")

    # Optional: display first spectrum comparison
    import matplotlib.pyplot as plt
    idx = 0
    plt.plot(wavelengths, spectra[idx], label='Original')
    plt.plot(wavelengths, recon_spectra[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Spectrum Reconstruction Example')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()