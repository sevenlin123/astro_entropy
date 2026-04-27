import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# Define wavelength grid and basis functions
# ----------------------------------------
def wavelength_grid(start_nm=400, end_nm=800, N=200):
    return np.linspace(start_nm, end_nm, N)

def basis_functions(wl):
    """Return an array of shape (N, n_basis)."""
    # 5 basis functions: constant, linear, quadratic, two Gaussians
    gauss1 = np.exp(-0.5 * ((wl - 500) / 50)**2)
    gauss2 = np.exp(-0.5 * ((wl - 650) / 30)**2)
    return np.vstack([np.ones_like(wl),
                      wl,
                      wl**2,
                      gauss1,
                      gauss2]).T

# ----------------------------------------
# Generate synthetic spectra
# ----------------------------------------
def generate_synthetic_spectrum(basis, coeffs):
    """basis: (N, n_basis), coeffs: (n_basis,)."""
    return basis @ coeffs

# ----------------------------------------
# Define simple top‑hat filter transmissions
# ----------------------------------------
def filters():
    """Return list of filter transmissions arrays (N,)."""
    wl = wavelength_grid()
    filt_blue  = np.logical_and(wl >= 400, wl < 500).astype(float)
    filt_green = np.logical_and(wl >= 500, wl < 600).astype(float)
    filt_red   = np.logical_and(wl >= 600, wl <= 700).astype(float)
    return [filt_blue, filt_green, filt_red]

# ----------------------------------------
# Compute photometry (integrated fluxes)
# ----------------------------------------
def photometry_from_spectrum(spectrum, filt, wl):
    """Integrate spectrum * filter over wavelength."""
    return simps(spectrum * filt, wl)

# ----------------------------------------
# Reconstruction routine
# ----------------------------------------
def reconstruct_spectrum(phot, basis, filt_list, wl):
    """
    phot: (n_filters,)
    basis: (N, n_basis)
    filt_list: list of (N,) filter transmissions
    wl: (N,) wavelength grid
    """
    # Build design matrix A: integrate each basis function with each filter
    A = np.zeros((len(filt_list), basis.shape[1]))
    for i, f in enumerate(filt_list):
        for j in range(basis.shape[1]):
            A[i, j] = simps(basis[:, j] * f, wl)
    # Solve least squares for coefficients
    coeff_est = np.linalg.lstsq(A, phot, rcond=None)[0]
    # Reconstruct spectrum
    return basis @ coeff_est

# ----------------------------------------
# Main demonstration
# ----------------------------------------
def main():
    np.random.seed(42)

    # Setup
    wl = wavelength_grid()
    B = basis_functions(wl)
    filt_list = filters()

    # True coefficients for synthetic spectrum
    coeff_true = np.array([1.0, -0.002, 1e-6, 3.0, 2.5])

    # Generate synthetic spectrum
    spec_true = generate_synthetic_spectrum(B, coeff_true)

    # Compute photometry
    phot = np.array([photometry_from_spectrum(spec_true, f, wl) for f in filt_list])

    # Reconstruct spectrum
    spec_rec = reconstruct_spectrum(phot, B, filt_list, wl)

    # Output results
    print("True coefficients:      ", coeff_true)
    print("Estimated coefficients:", np.linalg.lstsq(
        np.array([[simps(B[:,j]*f, wl) for f in filt_list] for j in range(B.shape[1])]).T,
        phot, rcond=None)[0])
    print("\nReconstruction error (RMS):", np.sqrt(np.mean((spec_true - spec_rec)**2)))

if __name__ == "__main__":
    main()