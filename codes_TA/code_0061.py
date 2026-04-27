import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ---------------------------------------------
# Define wavelength grid
wavelength = np.linspace(3000, 10000, 2000)  # in angstroms

# ---------------------------------------------
# Basis functions: Gaussian kernels
def gaussian_basis(wl, center, width):
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_basis(n_basis=10, wl=wavelength):
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    widths = (wl.max() - wl.min()) / (n_basis * 4)
    basis = np.array([gaussian_basis(wl, c, widths) for c in centers])
    return basis  # shape (n_basis, len(wl))

# ---------------------------------------------
# Generate synthetic spectra
def generate_spectra(n_samples, basis):
    amps = np.random.randn(n_samples, basis.shape[0])  # random amplitudes
    spectra = amps @ basis  # shape (n_samples, len(wl))
    return spectra, amps

# ---------------------------------------------
# Define simple top‑hat photometric filters
def build_filters(n_filters=5, wl=wavelength):
    filt_ranges = np.linspace(wl.min(), wl.max(), n_filters + 1)
    filters = []
    for i in range(n_filters):
        filt = np.zeros_like(wl)
        mask = (wl >= filt_ranges[i]) & (wl < filt_ranges[i+1])
        filt[mask] = 1.0
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, len(wl))

# ---------------------------------------------
# Compute photometry from spectra
def compute_photometry(spectra, filters):
    # spectra: (n_samples, len(wl))
    # filters: (n_filters, len(wl))
    return (spectra[:, None, :] * filters[None, :, :]).sum(axis=2)  # (n_samples, n_filters)

# ---------------------------------------------
# Construct design matrix (basis integrated over filters)
def construct_design_matrix(basis, filters):
    # basis: (n_basis, len(wl)), filters: (n_filters, len(wl))
    M = np.array([
        [simps(basis[j] * filters[i], wavelength) for j in range(basis.shape[0])]
        for i in range(filters.shape[0])
    ])  # shape (n_filters, n_basis)
    return M

# ---------------------------------------------
# Reconstruct spectrum from photometry
def reconstruct_spectrum(phot, design_matrix, wl=wavelength, alpha=1.0):
    # phot: (n_samples, n_filters), design_matrix: (n_filters, n_basis)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design_matrix.T, phot.T)  # learn basis coefficients
    coeffs = ridge.coef_.T  # shape (n_samples, n_basis)
    basis = build_basis(len(design_matrix[0]), wl)
    recon_spectra = coeffs @ basis  # shape (n_samples, len(wl))
    return recon_spectra, coeffs

# ---------------------------------------------
def main():
    np.random.seed(42)

    # Build basis and filters
    basis = build_basis(n_basis=15)
    filters = build_filters(n_filters=7)

    # Generate synthetic spectra
    spectra, true_coeffs = generate_spectra(n_samples=20, basis=basis)

    # Compute photometry
    phot = compute_photometry(spectra, filters)

    # Construct design matrix
    M = construct_design_matrix(basis, filters)

    # Reconstruct spectra
    recon_spectra, pred_coeffs = reconstruct_spectrum(phot, M)

    # Print some diagnostics
    print("True coeffs shape:", true_coeffs.shape)
    print("Predicted coeffs shape:", pred_coeffs.shape)
    print("Reconstructed spectra shape:", recon_spectra.shape)
    print("First reconstructed spectrum sample:")
    print(recon_spectra[0][:10])  # first 10 wavelength points

if __name__ == "__main__":
    main()