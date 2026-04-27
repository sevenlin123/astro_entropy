import numpy as np
from scipy.linalg import lstsq
from sklearn.linear_model import Ridge

# ----------------------- Spectral model ---------------------------------
def generate_synthetic_spectrum(wavelengths, center=5000., width=1000., amp=1.0):
    """
    Simple Gaussian spectral line on a flat continuum.
    """
    continuum = 0.5 * np.ones_like(wavelengths)
    line = amp * np.exp(-0.5 * ((wavelengths - center)/width)**2)
    return continuum + line

def generate_synthetic_spectra(n_spec, wavelengths, rng=None):
    rng = rng or np.random.default_rng()
    spectra = []
    for _ in range(n_spec):
        center = rng.uniform(4500, 5500)
        width  = rng.uniform(800, 1200)
        amp    = rng.uniform(0.8, 1.2)
        spectra.append(generate_synthetic_spectrum(wavelengths, center, width, amp))
    return np.array(spectra)

# ----------------------- Filters -----------------------------------------
def generate_filter_set(n_filters, wavelengths, rng=None):
    rng = rng or np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        # Randomly shape a top-hat filter
        low  = rng.uniform(wavelengths.min(), wavelengths.max() - 200)
        high = low + rng.uniform(100, 300)
        filt = np.zeros_like(wavelengths)
        filt[(wavelengths >= low) & (wavelengths <= high)] = 1.0
        filters.append(filt)
    return np.array(filters)

# ----------------------- Photometry --------------------------------------
def compute_photometry(spectra, filters, noise_sigma=0.01, rng=None):
    rng = rng or np.random.default_rng()
    # Integrate spectrum over each filter
    fluxes = spectra @ filters.T   # shape (n_spec, n_filters)
    # Add Gaussian noise
    noisy_fluxes = fluxes + rng.normal(0, noise_sigma, size=fluxes.shape)
    return noisy_fluxes

# ----------------------- Reconstruction ----------------------------------
def reconstruct_spectrum(photometry, filters, reg_strength=1e-4):
    """
    Solve linear system S @ F^T ≈ photometry for S.
    Use ridge regression to stabilise inversion.
    """
    # Transpose filters so shape is (n_filters, n_wavelengths)
    F = filters.T
    n_spec, n_filters = photometry.shape
    recon_spectra = np.empty((n_spec, F.shape[1]))
    for i in range(n_spec):
        y = photometry[i]
        # Ridge regression per spectrum
        ridge = Ridge(alpha=reg_strength, fit_intercept=False, solver="auto")
        ridge.fit(F, y)
        recon_spectra[i] = ridge.coef_
    return recon_spectra

# ----------------------- Example usage -----------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Wavelength grid
    wav = np.linspace(4000, 6000, 2000)

    # Generate synthetic spectra
    n_spectra = 5
    true_spectra = generate_synthetic_spectra(n_spectra, wav, rng)

    # Generate filter set
    n_filters = 10
    filter_bank = generate_filter_set(n_filters, wav, rng)

    # Compute photometry with noise
    phot = compute_photometry(true_spectra, filter_bank, noise_sigma=0.02, rng=rng)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectrum(phot, filter_bank, reg_strength=1e-2)

    # Simple comparison output
    for i in range(n_spectra):
        diff = np.linalg.norm(true_spectra[i] - recon_spectra[i])
        print(f"Spec {i}: reconstruction error norm = {diff:.4f}")