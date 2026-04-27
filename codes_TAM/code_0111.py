import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

# ---------------------------------------------
# 1. Spectral model (Planck function)
# ---------------------------------------------
def planck_lambda(wavelength_m, temperature):
    """Planck function B(λ,T) in W sr^-1 m^-3."""
    exponent = h * c / (wavelength_m * k * temperature)
    return (2.0 * h * c**2) / (wavelength_m**5 * (np.exp(exponent) - 1.0))

# ---------------------------------------------
# 2. Synthetic spectra generation
# ---------------------------------------------
def generate_basis_spectra(wl, temps):
    """Create a set of basis spectra (blackbodies)."""
    basis = []
    for t in temps:
        flux = planck_lambda(wl, t)
        basis.append(flux)
    return np.array(basis)   # shape (n_basis, n_wl)

def synth_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return np.dot(coeffs, basis)

# ---------------------------------------------
# 3. Photometric data generation
# ---------------------------------------------
def gaussian_filter(wl, center, width):
    """Simple Gaussian filter response."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

def photometric_flux(spectrum, wl, filt_wl, filt_resp):
    """
    Compute photometric flux for one filter:
    ∫ F(λ) R(λ) dλ / ∫ R(λ) dλ
    """
    interp_filt = np.interp(wl, filt_wl, filt_resp, left=0.0, right=0.0)
    num = trapz(spectrum * interp_filt, wl)
    denom = trapz(interp_filt, wl)
    return num / denom

def generate_photometry(spectrum, wl, filters):
    """
    Filters: list of dict with keys 'name', 'center', 'width'.
    Returns dict of fluxes.
    """
    phot = {}
    filt_wl = wl  # use same wavelength grid for simplicity
    for f in filters:
        resp = gaussian_filter(wl, f['center'], f['width'])
        phot[f['name']] = photometric_flux(spectrum, wl, filt_wl, resp)
    return phot

# ---------------------------------------------
# 4. Spectrum reconstruction from photometry
# ---------------------------------------------
def build_design_matrix(basis, wl, filters):
    """
    Build matrix A where A[i,j] = integrated response of basis j in filter i.
    Rows: filters, Columns: basis spectra.
    """
    filt_wl = wl
    n_filters = len(filters)
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i, f in enumerate(filters):
        resp = gaussian_filter(wl, f['center'], f['width'])
        for j in range(n_basis):
            # integrate basis[j] * resp
            A[i, j] = trapz(basis[j] * resp, wl) / trapz(resp, wl)
    return A

def reconstruct_spectrum_from_photometry(phot, wl, filters, basis):
    """
    Given photometric fluxes, wavelength grid, filters and basis set,
    solve for coefficients via Ridge regression (regularized least squares).
    """
    # Prepare target vector
    y = np.array([phot[f['name']] for f in filters])
    # Build design matrix
    A = build_design_matrix(basis, wl, filters)
    # Fit coefficients
    ridge = Ridge(alpha=1e-6, fit_intercept=False, solver='auto')
    ridge.fit(A, y)
    coeffs = ridge.coef_
    # Reconstruct spectrum
    recon = synth_spectrum(basis, coeffs)
    return recon, coeffs

# ---------------------------------------------
# 5. Demo
# ---------------------------------------------
def main():
    # Wavelength grid: 300 nm to 800 nm
    wl_nm = np.linspace(300, 800, 1000)          # nanometers
    wl_m  = wl_nm * 1e-9                         # meters

    # Basis spectra: blackbodies at 5000 K, 6000 K, 7000 K
    temps = [5000, 6000, 7000]
    basis = generate_basis_spectra(wl_m, temps)

    # Synthetic coefficients (unknown in real case)
    true_coeffs = np.array([0.3, 0.5, 0.2])
    true_spectrum = synth_spectrum(basis, true_coeffs)

    # Define photometric filters (Gaussian approximations)
    filters = [
        {'name': 'U', 'center': 365e-9, 'width': 30e-9},
        {'name': 'B', 'center': 445e-9, 'width': 40e-9},
        {'name': 'V', 'center': 551e-9, 'width': 40e-9},
        {'name': 'R', 'center': 658e-9, 'width': 50e-9},
        {'name': 'I', 'center': 806e-9, 'width': 70e-9},
    ]

    # Generate synthetic photometry
    phot = generate_photometry(true_spectrum, wl_m, filters)

    # Reconstruct spectrum from photometry
    recon_spectrum, recovered_coeffs = reconstruct_spectrum_from_photometry(
        phot, wl_m, filters, basis
    )

    # Print results
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs :", recovered_coeffs)

    # Optional: compute error metric
    mse = np.mean((true_spectrum - recon_spectrum)**2)
    print(f"Reconstruction MSE: {mse:.3e}")

if __name__ == "__main__":
    main()