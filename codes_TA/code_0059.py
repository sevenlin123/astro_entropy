#!/usr/bin/env python3
import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def wavelength_grid(npts=1000, lam_min=300, lam_max=2500):
    """Create a linear wavelength grid in nm."""
    return np.linspace(lam_min, lam_max, npts)

def gaussian_basis(wl, centers, widths):
    """
    Build a list of Gaussian basis functions evaluated on wl.
    Returns an array of shape (n_basis, len(wl)).
    """
    basis = []
    for c, w in zip(centers, widths):
        basis.append(np.exp(-0.5 * ((wl - c) / w)**2))
    return np.vstack(basis)

def generate_random_coeffs(n_samples, n_basis, rng=np.random.default_rng()):
    """Draw random coefficients for synthetic spectra."""
    return rng.normal(size=(n_samples, n_basis))

def synthesize_spectra(coeffs, basis):
    """Linear combination of basis functions."""
    # coeffs: (n_samples, n_basis)
    # basis: (n_basis, n_pixels)
    return coeffs @ basis

# ----------------------------------------------------------------------
# Photometric system
# ----------------------------------------------------------------------
def rectangular_filter(wl, center, width):
    """Simple top-hat filter centered at `center` with full width `width`."""
    return np.where((wl >= center - width/2) & (wl <= center + width/2), 1.0, 0.0)

def build_filters(wl, centers, widths):
    """Return a matrix of shape (n_filters, len(wl))."""
    filters = []
    for c, w in zip(centers, widths):
        filters.append(rectangular_filter(wl, c, w))
    return np.vstack(filters)

def compute_photometry(spectra, wl, filters):
    """
    Integrate spectra through each filter.
    spectra: (n_samples, n_pixels)
    filters: (n_filters, n_pixels)
    Returns fluxes: (n_samples, n_filters)
    """
    n_samples, _ = spectra.shape
    n_filters = filters.shape[0]
    flux = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        filt = filters[i]
        # Simple trapezoidal integration
        flux[:, i] = integrate.trapz(spectra * filt, wl, axis=1)
    return flux

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def construct_m_matrix(basis, filters):
    """
    Forward mapping matrix from coefficients to photometry:
    M_{ij} = ∫ B_j(λ) T_i(λ) dλ
    basis: (n_basis, n_pixels)
    filters: (n_filters, n_pixels)
    Returns M of shape (n_filters, n_basis)
    """
    n_basis = basis.shape[0]
    n_filters = filters.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            M[i, j] = integrate.trapz(basis[j] * filters[i], axis=0)
    return M

def reconstruct_coeffs(photon, M, reg_type="pseudoinv"):
    """
    Estimate coefficients from photometry.
    Options:
      - "pseudoinv": use Moore-Penrose inverse
      - "ridge": use Ridge regression
      - "linreg": use ordinary least squares
    """
    if reg_type == "pseudoinv":
        pinv_M = np.linalg.pinv(M)
        return photon @ pinv_M.T   # shape (n_samples, n_basis)
    elif reg_type == "ridge":
        lr = LinearRegression()
        lr.fit(M.T, np.eye(M.shape[0]))   # Fit M*c = I => c = M^+ * y
        return lr.predict(photon.T).T
    else:  # ordinary least squares
        lr = LinearRegression(fit_intercept=False)
        lr.fit(M.T, np.eye(M.shape[0]))
        return lr.predict(photon.T).T

# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # 1. Wavelength grid
    wl = wavelength_grid()

    # 2. Basis functions
    centers = [400, 600, 800, 1100, 1700]          # nm
    widths  = [50, 80, 120, 200, 300]              # nm
    basis = gaussian_basis(wl, centers, widths)     # (n_basis, n_pixels)

    # 3. Generate synthetic spectra
    n_samples = 100
    coeffs_true = generate_random_coeffs(n_samples, len(centers), rng)
    spectra = synthesize_spectra(coeffs_true, basis)   # (n_samples, n_pixels)

    # 4. Photometric filters
    filter_centers = [450, 650, 850, 1200, 1600]      # nm
    filter_widths  = [80, 120, 150, 200, 250]         # nm
    filters = build_filters(wl, filter_centers, filter_widths)  # (n_filters, n_pixels)

    # 5. Compute photometry
    phot = compute_photometry(spectra, wl, filters)    # (n_samples, n_filters)

    # 6. Reconstruction
    M = construct_m_matrix(basis, filters)             # (n_filters, n_basis)
    coeffs_rec = reconstruct_coeffs(phot, M, reg_type="pseudoinv")

    # 7. Reconstructed spectra
    spectra_rec = synthesize_spectra(coeffs_rec, basis)

    # 8. Evaluate accuracy
    rms_error = np.sqrt(np.mean((spectra - spectra_rec)**2))
    print(f"Reconstruction RMS error: {rms_error:.4f}")

if __name__ == "__main__":
    main()