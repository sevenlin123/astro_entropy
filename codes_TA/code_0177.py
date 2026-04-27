import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------- Spectral basis ----------
def create_basis(wl, n_basis):
    """Create Gaussian basis functions."""
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    widths = (wl.max() - wl.min()) / (2 * n_basis)
    basis = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths) ** 2)
    return basis  # shape (len(wl), n_basis)

# ---------- Bandpasses ----------
def gaussian_bandpass(wl, center, width):
    """Gaussian bandpass transmission."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def create_bandpasses(wl, n_band):
    """Generate bandpasses with random centers and widths."""
    rng = np.random.default_rng(seed=42)
    centers = rng.uniform(wl.min()+50, wl.max()-50, size=n_band)
    widths = rng.uniform(30, 80, size=n_band)
    bandpasses = [gaussian_bandpass(wl, c, w) for c, w in zip(centers, widths)]
    return bandpasses

# ---------- Synthetic spectra ----------
def synthesize_spectrum(basis, coeffs):
    """Generate spectrum as linear combination of basis."""
    return basis @ coeffs  # shape (len(wl),)

# ---------- Photometry ----------
def compute_photometry(spectrum, wl, bandpasses):
    """Integrate spectrum through each bandpass."""
    phot = []
    for bp in bandpasses:
        flux = np.trapz(spectrum * bp, wl)
        norm = np.trapz(bp, wl)
        phot.append(flux / norm)
    return np.array(phot)

# ---------- Bandpass matrix ----------
def bandpass_matrix(basis, wl, bandpasses):
    """Matrix M where M_ij = ∫ basis_j * bandpass_i / ∫ bandpass_i."""
    M = []
    for bp in bandpasses:
        integrals = [np.trapz(basis[:, j] * bp, wl) for j in range(basis.shape[1])]
        M.append(integrals / np.trapz(bp, wl))
    return np.vstack(M)  # shape (n_band, n_basis)

# ---------- Reconstruction ----------
def reconstruct_from_photometry(phot, M, alpha=1e-3):
    """Estimate coefficients by ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M, phot)
    return ridge.coef_

# ---------- Main ----------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 1000, 1000)  # nm

    # Basis functions
    n_basis = 5
    basis = create_basis(wl, n_basis)

    # Bandpasses
    n_band = 5
    bandpasses = create_bandpasses(wl, n_band)

    # Precompute bandpass matrix
    M = bandpass_matrix(basis, wl, bandpasses)

    # Generate synthetic data
    n_samples = 20
    rng = np.random.default_rng(seed=123)
    coeffs_true = rng.normal(size=(n_samples, n_basis))
    spectra = np.array([synthesize_spectrum(basis, c) for c in coeffs_true])
    photometry = np.array([compute_photometry(sp, wl, bandpasses) for sp in spectra])

    # Reconstruct the first spectrum
    idx = 0
    coeffs_est = reconstruct_from_photometry(photometry[idx], M, alpha=1e-3)
    spectrum_rec = synthesize_spectrum(basis, coeffs_est)

    # Compare
    mse = np.mean((spectra[idx] - spectrum_rec) ** 2)
    print(f"Reconstruction MSE for sample {idx}: {mse:.6f}")