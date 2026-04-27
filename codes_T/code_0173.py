import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Spectral model
# --------------------------------------------------------------------
def gaussian(wl, amp, cen, wid):
    """Single Gaussian."""
    return amp * np.exp(-(wl - cen)**2 / (2 * wid**2))

def synthetic_spectrum(wl, params):
    """
    Sum of Gaussian components.
    params : list of tuples [(amp, cen, wid), ...]
    """
    spec = np.zeros_like(wl)
    for amp, cen, wid in params:
        spec += gaussian(wl, amp, cen, wid)
    return spec

# --------------------------------------------------------------------
# Generate synthetic spectra
# --------------------------------------------------------------------
def generate_synthetic_spectra(n_stars, wl, n_components=3):
    """Create random spectra with random Gaussian parameters."""
    spectra = []
    true_params = []
    rng = np.random.default_rng()
    for _ in range(n_stars):
        params = []
        for _ in range(n_components):
            amp = rng.uniform(0.5, 1.5)
            cen = rng.uniform(5500, 6500)   # nm
            wid = rng.uniform(20, 50)       # nm
            params.append((amp, cen, wid))
        spec = synthetic_spectrum(wl, params)
        spectra.append(spec)
        true_params.append(params)
    return np.array(spectra), true_params

# --------------------------------------------------------------------
# Generate filter transmission curves
# --------------------------------------------------------------------
def generate_filters(n_filters, wl):
    """Random Gaussian filters."""
    rng = np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        cen = rng.uniform(5000, 7000)
        wid = rng.uniform(30, 100)
        filt = gaussian(wl, 1.0, cen, wid)
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return np.array(filters)

# --------------------------------------------------------------------
# Compute photometric fluxes
# --------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """Integrate spectrum*filter over wavelength."""
    return spectra @ filters.T  # shape: (n_stars, n_filters)

# --------------------------------------------------------------------
# Reconstruct spectra from photometry
# --------------------------------------------------------------------
def build_design_matrix(filters, wl, n_components):
    """
    For each filter, precompute its response to each Gaussian component
    with unit amplitude at fixed center positions.
    """
    # Use fixed centers for reconstruction basis
    rng = np.random.default_rng()
    centers = rng.uniform(5500, 6500, size=n_components)
    widths = np.linspace(20, 50, n_components)
    design = np.zeros((filters.shape[0], n_components))
    for i, filt in enumerate(filters):
        for j, (cen, wid) in enumerate(zip(centers, widths)):
            design[i, j] = gaussian(np.mean(wl), 1.0, cen, wid) @ filt
    return design, centers, widths

def reconstruct_spectra(photometry, filters, wl, n_components=3):
    """
    Fit linear combination of Gaussian basis functions to photometry.
    Returns estimated spectra.
    """
    design, centers, widths = build_design_matrix(filters, wl, n_components)
    # Solve least-squares for each star
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design, photometry.T)  # design: (n_filters, n_components)
    coeffs = reg.coef_.T          # shape: (n_stars, n_components)
    # Build spectra from coefficients
    recon = np.zeros((photometry.shape[0], len(wl)))
    for k in range(n_components):
        recon += coeffs[:, k][:, None] * gaussian(wl, 1.0, centers[k], widths[k])
    return recon, coeffs, centers, widths

# --------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (nm)
    wl = np.linspace(4000, 8000, 1000)

    # Synthetic data
    n_stars = 50
    spectra, true_params = generate_synthetic_spectra(n_stars, wl)
    n_filters = 5
    filters = generate_filters(n_filters, wl)

    # Photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruction
    recon_spectra, coeffs, centers, widths = reconstruct_spectra(
        photometry, filters, wl
    )

    # Simple assessment
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Reconstruction MSE: {mse:.4f}")