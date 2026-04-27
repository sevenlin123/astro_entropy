import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------- Configuration --------------------- #
WAVELENGTH_START = 300   # nm
WAVELENGTH_END   = 2500  # nm
WAVELENGTH_STEP  = 5     # nm
NUM_BASIS       = 10    # number of spectral basis functions
NUM_FILTERS     = 5     # number of photometric bands

np.random.seed(0)

# --------------------- Helper Functions --------------------- #
def gaussian(wl, center, width):
    """One-dimensional Gaussian."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_basis_functions():
    """Construct a set of Gaussian spectral basis functions."""
    wl = np.arange(WAVELENGTH_START, WAVELENGTH_END + WAVELENGTH_STEP, WAVELENGTH_STEP)
    centers = np.linspace(WAVELENGTH_START + 200, WAVELENGTH_END - 200, NUM_BASIS)
    width   = 80  # nm
    basis = np.vstack([gaussian(wl, c, width) for c in centers]).T  # shape (len(wl), NUM_BASIS)
    return wl, basis

def build_filter_curves():
    """Construct simple photometric filter transmission curves."""
    wl = np.arange(WAVELENGTH_START, WAVELENGTH_END + WAVELENGTH_STEP, WAVELENGTH_STEP)
    centers = np.linspace(WAVELENGTH_START + 150, WAVELENGTH_END - 150, NUM_FILTERS)
    width = 150  # nm
    filters = np.vstack([gaussian(wl, c, width) for c in centers]).T  # shape (len(wl), NUM_FILTERS)
    return wl, filters

def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs

def compute_photometry(spectrum, filters, wl):
    """Integrate spectrum weighted by each filter."""
    fluxes = np.array([np.trapz(spectrum * filt, wl) for filt in filters.T])
    return fluxes

def reconstruct_coefficients(photon, basis, filters, wl):
    """Fit basis coefficients from photometric measurements."""
    # Build design matrix: integral of filter * basis
    M = np.array([np.trapz(basis * filt[:, None], wl) for filt in filters.T])  # shape (NUM_FILTERS, NUM_BASIS)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M, photon)
    return reg.coef_

# --------------------- Main Workflow --------------------- #
if __name__ == "__main__":
    # Build basis functions
    wl, basis = build_basis_functions()

    # Build filter curves
    _, filters = build_filter_curves()

    # Generate a synthetic spectrum
    true_coeffs = np.random.randn(NUM_BASIS)
    true_spectrum = synthesize_spectrum(basis, true_coeffs)

    # Produce photometric observations (add small noise)
    photometry = compute_photometry(true_spectrum, filters, wl) + 0.02 * np.random.randn(NUM_FILTERS)

    # Reconstruct the spectrum
    recon_coeffs = reconstruct_coefficients(photometry, basis, filters, wl)
    recon_spectrum = synthesize_spectrum(basis, recon_coeffs)

    # Simple diagnostics
    error = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative reconstruction error: {error:.4f}")