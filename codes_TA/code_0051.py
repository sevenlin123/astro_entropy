import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Spectral basis ----------
def create_wavelength_grid(n=1000, lam_min=400, lam_max=800):
    """Wavelength grid in nm."""
    return np.linspace(lam_min, lam_max, n)

def gaussian_profile(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def spectral_basis(wavelength):
    """
    Return a list of basis functions evaluated on the wavelength grid.
    Here we use two Gaussian profiles.
    """
    b1 = gaussian_profile(wavelength, mu=500, sigma=50)
    b2 = gaussian_profile(wavelength, mu=700, sigma=30)
    return [b1, b2]

# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(num, wavelength, basis_funcs, rng=np.random.default_rng()):
    """
    Generate `num` synthetic spectra as random linear combinations of `basis_funcs`.
    Returns:
        spectra: array (num, len(wavelength))
        coeffs: array (num, len(basis_funcs))
    """
    coeffs = rng.uniform(low=0.5, high=2.0, size=(num, len(basis_funcs)))
    spectra = np.dot(coeffs, np.vstack(basis_funcs))
    return spectra, coeffs

# ---------- Photometric filters ----------
def create_filter_responses(wavelength):
    """
    Create a list of filter transmission curves (Gaussian-shaped).
    Returns:
        filters: list of arrays (len(wavelength))
    """
    f1 = gaussian_profile(wavelength, mu=450, sigma=40)
    f2 = gaussian_profile(wavelength, mu=550, sigma=40)
    f3 = gaussian_profile(wavelength, mu=650, sigma=40)
    return [f1, f2, f3]

# ---------- Photometry ----------
def compute_photometry(spectra, filters, wavelength):
    """
    Compute photometric fluxes for each spectrum through each filter.
    Uses Simpson's rule for integration.
    Returns:
        fluxes: array (num_spectra, num_filters)
    """
    fluxes = []
    for filt in filters:
        # Integrate S(lambda) * T(lambda) over lambda
        integrand = spectra * filt  # broadcasting over spectra rows
        flux = simps(integrand, wavelength, axis=1)  # integrate along wavelength axis
        fluxes.append(flux)
    return np.column_stack(fluxes)

# ---------- Reconstruction ----------
def reconstruct_spectra_from_photometry(fluxes, filter_integrals):
    """
    Reconstruct spectra coefficients from photometry.
    filter_integrals: array (num_filters, num_basis)
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(filter_integrals, fluxes.T)   # fit: coefficient matrix * coeffs = fluxes
    coeffs_rec = lr.coef_.T              # shape (num_spectra, num_basis)
    return coeffs_rec

def build_filter_integral_matrix(filters, basis_funcs, wavelength):
    """
    Build matrix A where A[j,i] = ∫ B_i(λ) * F_j(λ) dλ
    """
    num_filters = len(filters)
    num_basis = len(basis_funcs)
    A = np.zeros((num_filters, num_basis))
    for j, filt in enumerate(filters):
        for i, basis in enumerate(basis_funcs):
            integrand = basis * filt
            A[j, i] = simps(integrand, wavelength)
    return A

# ---------- Main demonstration ----------
if __name__ == "__main__":
    # 1. Define wavelength grid and basis
    wl = create_wavelength_grid()
    basis = spectral_basis(wl)

    # 2. Generate synthetic spectra
    num_spectra = 5
    spectra, true_coeffs = generate_synthetic_spectra(num_spectra, wl, basis)

    # 3. Create filter responses
    filters = create_filter_responses(wl)

    # 4. Compute photometry
    fluxes = compute_photometry(spectra, filters, wl)

    # 5. Build filter-integral matrix
    A = build_filter_integral_matrix(filters, basis, wl)

    # 6. Reconstruct spectra coefficients
    rec_coeffs = reconstruct_spectra_from_photometry(fluxes, A)

    # 7. Reconstruct spectra from recovered coefficients
    rec_spectra = np.dot(rec_coeffs, np.vstack(basis))

    # 8. Evaluate reconstruction error
    rel_errors = np.linalg.norm(rec_spectra - spectra, axis=1) / np.linalg.norm(spectra, axis=1)
    print("Relative reconstruction errors:", rel_errors)