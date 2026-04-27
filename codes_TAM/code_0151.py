import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. Spectral model – Gaussian basis functions
# ------------------------------------------------------------------
def make_gaussian_basis(n_basis=5, wl_min=4000, wl_max=8000, n_points=100):
    """
    Return wavelength grid and basis matrix (n_points x n_basis).
    Each column is a Gaussian centered at equally spaced wavelengths.
    """
    wl = np.linspace(wl_min, wl_max, n_points)
    centers = np.linspace(wl_min + 0.25*(wl_max-wl_min),
                          wl_max - 0.25*(wl_max-wl_min), n_basis)
    sigma = (wl_max-wl_min)/(n_basis*4)
    basis = np.exp(-0.5*((wl[:,None]-centers[None,:])/sigma)**2)
    return wl, basis

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, basis, rng=None):
    """
    Randomly generate spectra as linear combinations of the basis.
    """
    rng = rng or np.random.default_rng()
    coeffs = rng.uniform(0.5, 2.0, size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

# ------------------------------------------------------------------
# 3. Define filter transmission curves
# ------------------------------------------------------------------
def make_filters(filter_centers, filter_width, wl):
    """
    Return list of filter transmission functions (as arrays over wl).
    """
    filters = []
    for fc in filter_centers:
        filt = np.exp(-0.5*((wl-fc)/filter_width)**2)
        filters.append(filt)
    return filters

# ------------------------------------------------------------------
# 4. Compute photometry (flux in each filter)
# ------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate each spectrum through each filter.
    Returns array shape (n_spectra, n_filters).
    """
    n_filters = len(filters)
    fluxes = np.empty((spectra.shape[0], n_filters))
    for i, filt in enumerate(filters):
        norm = simps(filt, axis=0)
        fluxes[:,i] = simps(spectra * filt, axis=1) / norm
    return fluxes

# ------------------------------------------------------------------
# 5. Reconstruct spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_from_photometry(fluxes, basis, filters):
    """
    Solve for basis coefficients that best reproduce the photometry.
    Returns reconstructed spectra (n_samples x n_wavelengths).
    """
    # Build design matrix: integrate basis*filter
    n_filters = len(filters)
    design = np.zeros((n_filters, basis.shape[1]))
    for i, filt in enumerate(filters):
        design[i] = simps(basis * filt, axis=0) / simps(filt, axis=0)
    # Fit coefficients per spectrum
    reg = LinearRegression(fit_intercept=False).fit(design, fluxes.T)
    coeffs_rec = reg.coef_.T  # shape (n_samples, n_basis)
    rec_spectra = coeffs_rec @ basis.T
    return rec_spectra

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    n_samples   = 10
    n_basis     = 5
    wl_min      = 4000
    wl_max      = 8000
    n_points    = 200
    rng         = np.random.default_rng(42)

    # 1. Build basis
    wl, basis = make_gaussian_basis(n_basis=n_basis,
                                    wl_min=wl_min,
                                    wl_max=wl_max,
                                    n_points=n_points)

    # 2. Generate spectra
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis, rng=rng)

    # 3. Define filters
    filter_centers = [4700, 5200, 5700]
    filter_width   = 200
    filters = make_filters(filter_centers, filter_width, wl)

    # 4. Compute photometry
    fluxes = compute_photometry(spectra, filters)

    # 5. Reconstruct spectra
    recon_spectra = reconstruct_from_photometry(fluxes, basis, filters)

    # Simple check: print first true vs reconstructed spectrum
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra[idx], label='True')
    plt.plot(wl, recon_spectra[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum reconstruction example')
    plt.legend()
    plt.tight_layout()
    plt.show()