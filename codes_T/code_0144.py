import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# 1. Spectral model
# --------------------------------------------------------------------------- #

def make_wavelength_grid(n_wave, wl_start=400.0, wl_end=800.0):
    """Uniform wavelength grid in nm."""
    return np.linspace(wl_start, wl_end, n_wave)

def build_basis(n_basis, wavelengths, sigma=20.0):
    """
    Build a set of Gaussian basis functions evenly spaced over the wavelength range.
    Each basis function is normalized.
    """
    n_wave = len(wavelengths)
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c)/sigma)**2)
        basis.append(g / np.linalg.norm(g))
    return np.vstack(basis)      # shape (n_basis, n_wave)

# --------------------------------------------------------------------------- #
# 2. Synthetic spectra generation
# --------------------------------------------------------------------------- #

def generate_synthetic_spectra(n_obj, basis, noise_std=0.02, rng=None):
    """
    Generate synthetic spectra as random linear combinations of basis functions
    with added Gaussian noise.
    """
    rng = np.random.default_rng(rng)
    n_basis, n_wave = basis.shape
    weights = rng.uniform(low=-1.0, high=1.0, size=(n_obj, n_basis))
    spectra = weights @ basis          # (n_obj, n_wave)
    noise = rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, weights

# --------------------------------------------------------------------------- #
# 3. Photometric filter construction
# --------------------------------------------------------------------------- #

def make_gaussian_filter(wavelengths, center, width, amplitude=1.0):
    """Create a single Gaussian filter."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center)/width)**2)

def build_filters(n_filters, wavelengths, rng=None):
    """
    Create a set of Gaussian filters randomly positioned across the wavelength range.
    Returns a list of filter transmission curves (length n_filters).
    """
    rng = np.random.default_rng(rng)
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    centers = rng.uniform(low=wl_min, high=wl_max, size=n_filters)
    widths  = rng.uniform(low=(wl_max-wl_min)/20, high=(wl_max-wl_min)/10, size=n_filters)
    filters = [make_gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)           # shape (n_filters, n_wave)

# --------------------------------------------------------------------------- #
# 4. Photometry simulation
# --------------------------------------------------------------------------- #

def compute_photometry(spectra, filters):
    """
    Compute synthetic broadband fluxes by integrating spectra with each filter.
    """
    # spectra: (n_obj, n_wave), filters: (n_filters, n_wave)
    return spectra @ filters.T         # (n_obj, n_filters)

# --------------------------------------------------------------------------- #
# 5. Spectrum reconstruction from photometry
# --------------------------------------------------------------------------- #

def reconstruct_spectra(photometry, filters, basis, alpha=1e-3):
    """
    Reconstruct spectra from broadband photometry using linear regression.
    Solves for basis weights for each object and returns reconstructed spectra.
    """
    # Build the design matrix A: each filter -> dot product with each basis function
    # A shape: (n_filters, n_basis)
    A = filters @ basis.T              # (n_filters, n_wave) @ (n_wave, n_basis)
    
    # Use Ridge regression to obtain pseudo-inverse of A
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, np.eye(A.shape[0]))   # fit on identity to get pseudoinverse
    A_inv = ridge.coef_.T              # shape (n_basis, n_filters)
    
    # Estimate weights for each object
    weights_est = photometry @ A_inv   # (n_obj, n_filters) @ (n_filters, n_basis)
    reconstructed = weights_est @ basis   # (n_obj, n_wave)
    return reconstructed, weights_est

# --------------------------------------------------------------------------- #
# 6. Main routine
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)

    # Parameters
    n_wave  = 500            # number of wavelength points
    n_basis = 15             # number of spectral basis functions
    n_obj   = 100            # number of synthetic objects
    n_filt  = 12             # number of photometric filters

    # Build components
    wavelengths = make_wavelength_grid(n_wave)
    basis       = build_basis(n_basis, wavelengths)
    filters     = build_filters(n_filt, wavelengths, rng=rng)

    # Generate synthetic data
    spectra, true_weights = generate_synthetic_spectra(n_obj, basis, rng=rng)
    photometry          = compute_photometry(spectra, filters)

    # Reconstruction
    recon_spectra, est_weights = reconstruct_spectra(photometry, filters, basis)

    # Evaluation
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.6f}")

    # Show example spectra (optional, no plots required)
    idx = 0
    print("\nTrue spectrum (first 10 values):")
    print(spectra[idx, :10])
    print("Reconstructed spectrum (first 10 values):")
    print(recon_spectra[idx, :10])