import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Spectral model – basis functions
# ----------------------------------------------------------------------
def build_basis(wl, n_lines=5, width=10):
    """Return a matrix of basis spectra (Gaussian lines)."""
    N = len(wl)
    basis = np.zeros((N, n_lines))
    centers = np.linspace(wl[0] + (wl[-1]-wl[0])/(n_lines+1),
                          wl[-1] - (wl[-1]-wl[0])/(n_lines+1), n_lines)
    for i, c in enumerate(centers):
        # Gaussian line: amplitude=1, sigma=width
        basis[:, i] = np.exp(-0.5*((wl - c)/width)**2)
    return basis

# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis, n_samples=50, noise_std=0.05):
    """Generate synthetic spectra as linear combinations of basis."""
    n_basis = basis.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = basis @ coeffs.T   # shape (n_pixels, n_samples)
    spectra += noise_std * np.random.randn(*spectra.shape)
    return spectra.T, coeffs  # spectra: (n_samples, n_pixels)

# ----------------------------------------------------------------------
# 3. Filter (photometric) generation
# ----------------------------------------------------------------------
def build_filters(wl, n_filters=3):
    """Return filter response matrices (top‑hat)."""
    N = len(wl)
    filters = np.zeros((N, n_filters))
    band_width = (wl[-1] - wl[0]) / (n_filters * 2)
    centers = np.linspace(wl[0] + band_width,
                          wl[-1] - band_width, n_filters)
    for i, c in enumerate(centers):
        mask = (wl >= c - band_width/2) & (wl <= c + band_width/2)
        filters[mask, i] = 1.0
    return filters

# ----------------------------------------------------------------------
# 4. Photometry computation
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters, wl):
    """
    Integrate spectra over each filter.
    spectra: (n_samples, n_pixels)
    filters: (n_pixels, n_filters)
    Return photometric fluxes: (n_samples, n_filters)
    """
    # Normalise filters by their integral
    norm = np.sum(filters, axis=0)[None, :]
    flux = spectra @ filters / norm
    return flux

# ----------------------------------------------------------------------
# 5. Spectrum reconstruction from photometry
# ----------------------------------------------------------------------
def reconstruct_spectra(photometry, filters, basis, wl, alpha=1e-3):
    """
    Solve for coefficients that best reproduce the photometry.
    Uses Ridge regression (least‑squares with small regularisation).
    Returns reconstructed spectra.
    """
    # Build the design matrix M: integral of basis*filter over wl
    M = np.zeros((filters.shape[1], basis.shape[1]))
    for j in range(filters.shape[1]):
        filt = filters[:, j][:, None]         # (n_pixels, 1)
        M[j, :] = np.trapz(basis * filt, wl, axis=0)
    # Solve M * coeffs = photometry  (coeffs are (n_filters, n_samples))
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M, photometry.T)
    coeffs = ridge.coef_.T  # shape (n_samples, n_filters)
    # Reconstruct spectra
    spectra_rec = coeffs @ M  # shape (n_samples, n_filters) -> need to map back?
    # Actually we need to reconstruct using the basis and the fitted coefficients for each sample
    coeffs_sample = ridge.predict(M)  # not needed
    # Instead perform least-squares for each sample individually
    rec_spectra = []
    for s_flux in photometry:
        # Solve M * coeffs = s_flux
        coeffs_sol, *_ = np.linalg.lstsq(M, s_flux, rcond=None)
        rec_spectra.append(basis @ coeffs_sol)
    return np.array(rec_spectra)

# ----------------------------------------------------------------------
# 6. Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(4000, 7000, 1000)  # Angstrom

    # Build basis, filters
    basis = build_basis(wl, n_lines=5, width=20)
    filters = build_filters(wl, n_filters=3)

    # Generate training data
    spectra_true, true_coeffs = generate_synthetic_spectra(basis, n_samples=200, noise_std=0.02)
    photometry = compute_photometry(spectra_true, filters, wl)

    # Reconstruct spectra from photometry
    spectra_rec = reconstruct_spectra(photometry, filters, basis, wl, alpha=1e-2)

    # Evaluate reconstruction
    error = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {error:.6f}")

    # Visual check for one example
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra_true[idx], label='True spectrum')
    plt.plot(wl, spectra_rec[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.show()