import numpy as np
from sklearn.linear_model import LinearRegression

def make_wavelength_grid(start=300, stop=2500, points=1000):
    """Create a linear wavelength grid in nm."""
    return np.linspace(start, stop, points)

def gaussian(wl, mu, sigma):
    """1‑D Gaussian."""
    return np.exp(-0.5 * ((wl - mu) / sigma)**2)

def create_basis_spectra(n_basis, wl):
    """Generate n_basis synthetic basis spectra."""
    mus = np.linspace(wl.min() + 0.1*wl.ptp(),
                      wl.max() - 0.1*wl.ptp(), n_basis)
    sigmas = 0.05 * wl.ptp() * np.ones(n_basis)
    basis = np.array([gaussian(wl, m, s) for m, s in zip(mus, sigmas)])
    # normalize each basis to unit area
    basis /= basis.sum(axis=1, keepdims=True)
    return basis

def synthesize_spectra(n_samples, basis, noise_frac=0.02):
    """Generate synthetic spectra as random combinations of basis."""
    coeffs = np.random.rand(n_samples, basis.shape[0])
    spectra = coeffs @ basis
    # add small Gaussian noise
    noise = noise_frac * np.random.randn(*spectra.shape) * spectra.std(axis=1, keepdims=True)
    return spectra, coeffs

def create_filter(name, wl, center, width):
    """Top‑hat filter transmission curve."""
    trans = np.where(np.abs(wl - center) <= width/2, 1.0, 0.0)
    return trans

def generate_filters(names_centers_widths, wl):
    """Create dictionary of filter transmission curves."""
    return {name: create_filter(name, wl, cen, wid) for name, cen, wid in names_centers_widths}

def compute_photometry(spectra, filters, wl):
    """Integrate spectra through filters to obtain synthetic photometry."""
    dw = wl[1] - wl[0]
    fluxes = []
    for filt_name, filt in filters.items():
        flux = spectra @ (filt * dw)
        fluxes.append(flux)
    return np.vstack(fluxes).T   # shape (n_samples, n_filters)

def reconstruct_coefficients(photon, filters, basis, wl):
    """
    Solve for basis coefficients that reproduce the photometric fluxes.
    Photon: array (n_samples, n_filters)
    Returns: coeffs array (n_samples, n_basis)
    """
    dw = wl[1] - wl[0]
    n_filters = len(filters)
    n_basis = basis.shape[0]
    # Build design matrix P (n_filters, n_basis)
    P = np.zeros((n_filters, n_basis))
    for i, (name, filt) in enumerate(filters.items()):
        for j in range(n_basis):
            P[i, j] = np.sum(basis[j] * filt * dw)
    # Solve linear system for each sample
    coeffs = []
    lr = LinearRegression(fit_intercept=False).fit(P, photon.T)
    coeffs = lr.predict(P).T  # shape (n_samples, n_filters)
    # Since we solved P * coeffs' = photon, coeffs' is transposed
    coeffs = coeffs.T
    # Now adjust: we actually want coeffs such that photon = P @ coeffs^T
    # So coefficients are transposed of what LR gave
    return coeffs

def reconstruct_spectrum(coeffs, basis):
    """Rebuild spectra from basis coefficients."""
    return coeffs @ basis

def main():
    # 1. Create wavelength grid
    wl = make_wavelength_grid()
    # 2. Create basis spectra
    basis = create_basis_spectra(n_basis=5, wl=wl)
    # 3. Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = synthesize_spectra(n_samples, basis)
    # 4. Define filters
    filter_defs = [
        ("u", 360, 80),
        ("g", 480, 120),
        ("r", 620, 140)
    ]
    filters = generate_filters(filter_defs, wl)
    # 5. Compute synthetic photometry
    phot = compute_photometry(spectra, filters, wl)
    # 6. Reconstruct spectra from photometry
    # Build design matrix once
    dw = wl[1]-wl[0]
    n_filters = len(filters)
    n_basis = basis.shape[0]
    P = np.zeros((n_filters, n_basis))
    for i, (name, filt) in enumerate(filters.items()):
        for j in range(n_basis):
            P[i, j] = np.sum(basis[j] * filt * dw)
    # Solve for coefficients using least squares per sample
    coeffs_est = np.linalg.lstsq(P, phot.T, rcond=None)[0].T
    # 7. Reconstruct spectra
    recon_spectra = reconstruct_spectrum(coeffs_est, basis)
    # 8. Compare true vs reconstructed
    print("True coefficients (first sample):", true_coeffs[0])
    print("Estimated coefficients (first sample):", coeffs_est[0])
    print("Reconstruction error (RMS):",
          np.sqrt(((spectra - recon_spectra)**2).mean()))

if __name__ == "__main__":
    main()