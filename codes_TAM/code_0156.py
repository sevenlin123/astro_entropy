import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model and utilities
# ----------------------------------------------------------------------
def create_wavelength_grid(wl_min=400, wl_max=1000, n_points=200):
    return np.linspace(wl_min, wl_max, n_points)

def create_basis_spectra(n_basis, n_wave, rng=None):
    rng = np.random.default_rng(rng)
    # random smooth spectra
    return rng.normal(size=(n_basis, n_wave))  # simple random spectra

def generate_synthetic_spectra(n_samples, basis, noise_std=0.05, rng=None):
    rng = np.random.default_rng(rng)
    n_basis, n_wave = basis.shape
    coeffs = rng.uniform(low=-1, high=1, size=(n_samples, n_basis))
    spectra = coeffs @ basis  # linear combination
    noise = rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

# ----------------------------------------------------------------------
# Filter definition and photometry
# ----------------------------------------------------------------------
def gaussian_filter(wavelength, center, width):
    return np.exp(-0.5 * ((wavelength - center) / width) ** 2)

def create_filters(filter_params, wavelength):
    # filter_params: list of (center, width) tuples
    filters = [gaussian_filter(wavelength, c, w) for c, w in filter_params]
    return filters

def compute_photometry(spectra, filters, wavelength):
    # integrate spectrum * filter over wavelength
    phot = []
    for filt in filters:
        flux = simps(spectra * filt[None, :], x=wavelength, axis=1)
        phot.append(flux)
    return np.vstack(phot).T  # shape (n_samples, n_filters)

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def build_design_matrix(basis, filters, wavelength):
    # compute integrated response of each basis spectrum in each filter
    design = []
    for filt in filters:
        col = simps(basis.T * filt, x=wavelength, axis=1)  # shape (n_basis,)
        design.append(col)
    return np.vstack(design).T  # shape (n_filters, n_basis)

def reconstruct_coefficients(photometry, design_matrix):
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design_matrix.T, photometry.T)
    return lr.coef_.T  # shape (n_samples, n_basis)

def reconstruct_spectra(coefficients, basis):
    return coefficients @ basis

# ----------------------------------------------------------------------
# Main synthetic experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng_seed = 42
    wl = create_wavelength_grid()
    n_basis = 5
    basis = create_basis_spectra(n_basis, len(wl), rng=rng_seed)

    n_samples = 20
    spectra, true_coeffs = generate_synthetic_spectra(
        n_samples, basis, noise_std=0.02, rng=rng_seed
    )

    filter_params = [(450, 30), (550, 40), (650, 35)]
    filters = create_filters(filter_params, wl)

    phot = compute_photometry(spectra, filters, wl)
    design = build_design_matrix(basis, filters, wl)
    recon_coeffs = reconstruct_coefficients(phot, design)
    recon_spectra = reconstruct_spectra(recon_coeffs, basis)

    # Simple diagnostics
    mse = np.mean((spectra - recon_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.6f}")
    print("First spectrum: true vs reconstructed (first 5 values)")
    print(spectra[0, :5])
    print(recon_spectra[0, :5])