import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Model ----------
def gaussian_basis(lam, centers, width=0.05):
    """Return Gaussian basis functions evaluated at wavelengths lam."""
    return np.exp(-0.5 * ((lam[:, None] - centers[None, :]) / width)**2)

def build_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs

# ---------- Filters ----------
def rect_filter(lambda_min, lambda_max, lam):
    """Return boolean mask for a rectangular filter."""
    return (lam >= lambda_min) & (lam <= lambda_max)

def filter_matrix(basis, filters, lam):
    """
    Build matrix that maps coefficients to photometric fluxes.
    Each row corresponds to a filter, each column to a basis function.
    """
    mat = []
    for filt in filters:
        mask = rect_filter(*filt, lam)
        # integrate each basis function over the filter band
        integrals = [simps(b[mask], lam[mask]) for b in basis.T]
        mat.append(integrals)
    return np.array(mat)

def measure_photometry(spectrum, filters, lam):
    """Compute photometric measurements for a spectrum."""
    fluxes = []
    for filt in filters:
        mask = rect_filter(*filt, lam)
        flux = simps(spectrum[mask], lam[mask])
        fluxes.append(flux)
    return np.array(fluxes)

# ---------- Reconstruction ----------
def reconstruct_coeffs(photon, filter_mat):
    """Recover coefficients via ordinary least squares."""
    lr = LinearRegression(fit_intercept=False).fit(filter_mat, photon)
    return lr.coef_

# ---------- Demo ----------
def main():
    # Wavelength grid
    lam = np.linspace(0.4, 1.0, 200)   # microns

    # Basis
    n_basis = 10
    centers = np.linspace(0.4, 1.0, n_basis)
    basis = gaussian_basis(lam, centers)

    # Filters (start, end) in microns
    filters = [(0.45, 0.55), (0.60, 0.70), (0.80, 0.90)]

    # Build filter matrix
    F = filter_matrix(basis, filters, lam)

    # Synthetic spectrum
    true_coeffs = np.random.rand(n_basis)
    spec = build_spectrum(basis, true_coeffs)

    # Photometry
    phot = measure_photometry(spec, filters, lam)

    # Reconstruction
    rec_coeffs = reconstruct_coeffs(phot, F)
    rec_spec = build_spectrum(basis, rec_coeffs)

    # Print results
    print("True coeffs:", true_coeffs)
    print("Reconstructed coeffs:", rec_coeffs)
    print("Norm error:", np.linalg.norm(true_coeffs - rec_coeffs))

if __name__ == "__main__":
    main()