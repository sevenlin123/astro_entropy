import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# --------------------- Spectral model --------------------- #
def gaussian_basis(wl, centers, widths):
    """Generate a matrix of Gaussian basis functions."""
    basis = []
    for c, w in zip(centers, widths):
        basis.append(np.exp(-0.5 * ((wl - c) / w) ** 2))
    return np.vstack(basis).T  # shape: (len(wl), len(centers))

def synth_spectra(n_samples, wl, basis):
    """Create synthetic spectra as random linear combos of basis functions."""
    coeffs = np.random.randn(n_samples, basis.shape[1])
    spectra = coeffs @ basis.T  # shape: (n_samples, len(wl))
    return spectra, coeffs

# --------------------- Photometry --------------------- #
def filter_gaussian(wl, center, width):
    """Filter transmission curve: Gaussian."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def photometric_vector(spectrum, wl, filters):
    """Integrate spectrum through each filter to get photometric measurements."""
    vec = []
    for filt in filters:
        trans = filter_gaussian(wl, *filt)
        flux = simps(spectrum * trans, wl) / simps(trans, wl)
        vec.append(flux)
    return np.array(vec)

def build_filter_matrix(basis, wl, filters):
    """Compute how each filter measures each basis function."""
    F = []
    for filt in filters:
        trans = filter_gaussian(wl, *filt)
        # integrate basis*k * trans
        fluxes = [simps(b * trans, wl) / simps(trans, wl) for b in basis.T]
        F.append(fluxes)
    return np.array(F).T  # shape: (len(basis), len(filters))

# --------------------- Reconstruction --------------------- #
def reconstruct_spectra(photometry, filter_matrix, wl, n_components=50):
    """Estimate coefficients via Ridge regression and reconstruct spectra."""
    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(filter_matrix.T, photometry.T)  # regress each basis coefficient
    coeffs_hat = reg.coef_.T  # shape: (n_samples, n_basis)
    spectra_hat = coeffs_hat @ basis.T
    return spectra_hat, coeffs_hat

# --------------------- Main --------------------- #
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 800, 81)  # nm

    # Basis functions
    num_basis = 20
    centers = np.linspace(420, 780, num_basis)
    widths = np.full(num_basis, 20.0)  # nm
    basis = gaussian_basis(wl, centers, widths)  # shape: (len(wl), num_basis)

    # Generate synthetic spectra
    n_samples = 15
    spectra_true, coeffs_true = synth_spectra(n_samples, wl, basis)

    # Define filters: (center, width)
    filters = [(440, 30), (550, 30), (660, 30)]  # U, B, V

    # Build filter matrix
    filter_mat = build_filter_matrix(basis, wl, filters)  # shape: (num_basis, n_filters)

    # Generate photometry with small noise
    photometry = []
    for i in range(n_samples):
        vec = photometric_vector(spectra_true[i], wl, filters)
        vec += np.random.normal(scale=0.01, size=vec.shape)  # noise
        photometry.append(vec)
    photometry = np.array(photometry)  # shape: (n_samples, n_filters)

    # Reconstruction
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filter_mat, wl)

    # Evaluate reconstruction
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error of reconstructed spectra: {mse:.4e}")

    # Show true vs recovered coefficients for first sample
    print("\nTrue coefficients (first sample):")
    print(coeffs_true[0])
    print("\nRecovered coefficients (first sample):")
    print(coeffs_rec[0])