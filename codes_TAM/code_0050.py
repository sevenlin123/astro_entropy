import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Define a simple spectral model
# ----------------------------------------------------------------------
def generate_basis(num_wavelengths: int, num_bases: int, seed: int = 42):
    """
    Generate a set of orthogonal-ish basis spectra.
    Each basis is a Gaussian bump at a random centre wavelength.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(300, 2500, num_wavelengths)  # nm
    bases = []
    for _ in range(num_bases):
        center = rng.uniform(400, 2400)
        width = rng.uniform(50, 200)
        amp = rng.uniform(0.5, 1.5)
        spec = amp * np.exp(-0.5 * ((wl - center) / width) ** 2)
        bases.append(spec)
    return np.vstack(bases)  # shape (num_bases, num_wavelengths)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def synthesize_spectra(basis: np.ndarray, n_samples: int, noise_level: float = 0.02, seed: int = 24):
    """
    Create synthetic spectra by drawing random coefficients for each basis.
    Add Gaussian noise to the resulting spectra.
    """
    rng = np.random.default_rng(seed)
    n_bases, n_wl = basis.shape
    coeffs = rng.uniform(0.2, 1.0, size=(n_samples, n_bases))
    spectra = coeffs @ basis  # shape (n_samples, n_wl)
    noise = rng.normal(scale=noise_level, size=spectra.shape)
    return spectra, coeffs, wl  # return wavelengths for reference

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------------
def create_filters(num_filters: int, wl: np.ndarray, seed: int = 99):
    """Create simple top-hat filters covering different wavelength ranges."""
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(num_filters):
        start = rng.uniform(wl[0], wl[-1] - 150)
        end = start + rng.uniform(80, 200)
        filt = np.where((wl >= start) & (wl <= end), 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)  # shape (num_filters, n_wl)

def photometric_observations(spectra: np.ndarray, filters: np.ndarray):
    """
    Integrate each spectrum through each filter to obtain fluxes.
    """
    # Normalize filter responses to unit area for simplicity
    filt_norm = filters / simps(filters, axis=1, dx=1)
    fluxes = spectra @ filt_norm.T  # shape (n_samples, num_filters)
    return fluxes

# ----------------------------------------------------------------------
# 4. Reconstruct a synthetic spectrum from photometric data
# ----------------------------------------------------------------------
def build_design_matrix(basis: np.ndarray, filters: np.ndarray):
    """
    Compute how each basis contributes to each filter.
    """
    filt_norm = filters / simps(filters, axis=1, dx=1)
    # basis shape (n_bases, n_wl), filt_norm shape (n_filters, n_wl)
    # result shape (n_filters, n_bases)
    return filt_norm @ basis.T

def reconstruct_coefficients(fluxes: np.ndarray, design: np.ndarray, alpha: float = 1e-4):
    """
    Solve for basis coefficients given photometric fluxes.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(design, fluxes.T)  # design: (filters, bases), fluxes: (samples, filters)
    return reg.coef_.T  # shape (samples, bases)

def reconstruct_spectra(coeffs: np.ndarray, basis: np.ndarray):
    """Combine basis spectra with recovered coefficients."""
    return coeffs @ basis  # shape (samples, n_wl)

# ----------------------------------------------------------------------
# Demo execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    N_WL = 1000          # number of wavelength points
    N_BASIS = 5          # number of basis spectra
    N_SAMPLES = 20       # number of synthetic objects
    N_FILTERS = 4        # number of photometric filters

    # Step 1: Basis spectra
    basis = generate_basis(N_WL, N_BASIS)

    # Step 2: Synthetic spectra
    spectra, true_coeffs, wl = synthesize_spectra(basis, N_SAMPLES)

    # Step 3: Photometric data
    filters = create_filters(N_FILTERS, wl)
    fluxes = photometric_observations(spectra, filters)

    # Step 4: Reconstruction
    design_mat = build_design_matrix(basis, filters)
    recovered_coeffs = reconstruct_coefficients(fluxes, design_mat)
    recovered_spectra = reconstruct_spectra(recovered_coeffs, basis)

    # Evaluate reconstruction accuracy
    err = np.mean((recovered_spectra - spectra) ** 2, axis=1)
    print("Mean squared error per spectrum:", err)
    print("Overall mean squared error:", err.mean())