import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model: a sum of Gaussian basis functions
# ------------------------------------------------------------------
def create_basis_functions(num_funcs, wl):
    """
    Create `num_funcs` Gaussian basis functions over wavelength grid `wl`.
    Returns an array of shape (num_funcs, len(wl)).
    """
    centers = np.linspace(wl[0] + 0.1*(wl[-1]-wl[0]),
                          wl[-1] - 0.1*(wl[-1]-wl[0]),
                          num_funcs)
    std = (wl[-1] - wl[0])/(4*num_funcs)   # width of each Gaussian
    basis = np.array([norm.pdf(wl, loc=c, scale=std) for c in centers])
    # Normalize each basis function to unit integral
    basis /= np.trapz(basis, wl, axis=1)[:, None]
    return basis

# ------------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis_funcs, rng=None):
    """
    Generate `n_samples` spectra as linear combinations of `basis_funcs`.
    Coefficients are drawn from a normal distribution.
    """
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, basis_funcs.shape[0]))
    spectra = coeffs @ basis_funcs
    return spectra, coeffs

# ------------------------------------------------------------------
# Define filter transmission curves (Gaussians)
# ------------------------------------------------------------------
def create_filters(n_filters, wl):
    """
    Create `n_filters` Gaussian filters over wavelength grid `wl`.
    Returns an array of shape (n_filters, len(wl)).
    """
    centers = np.linspace(wl[0] + 0.15*(wl[-1]-wl[0]),
                          wl[-1] - 0.15*(wl[-1]-wl[0]),
                          n_filters)
    std = (wl[-1] - wl[0])/(8*n_filters)
    filters = np.array([norm.pdf(wl, loc=c, scale=std) for c in centers])
    # Normalize each filter to have unit integral (for photometric scaling)
    filters /= np.trapz(filters, wl, axis=1)[:, None]
    return filters

# ------------------------------------------------------------------
# Compute photometric measurements
# ------------------------------------------------------------------
def compute_photometry(spectra, filters, wl):
    """
    Integrate each spectrum through each filter to produce photometric
    magnitudes. Returns an array of shape (n_samples, n_filters).
    """
    # Using trapezoidal integration over wavelength
    integrals = spectra @ (filters.T * wl[None, :, None])
    # Divide by filter integrals (already normalized)
    phot = integrals
    return phot

# ------------------------------------------------------------------
# Reconstruction: recover coefficients from photometry
# ------------------------------------------------------------------
def build_design_matrix(filters, basis_funcs, wl):
    """
    For each filter, compute its effect on each basis function.
    Returns a matrix of shape (n_filters, n_basis).
    """
    # Integral of (basis * filter) over wl
    integrals = basis_funcs @ (filters.T * wl[None, :, None])
    # Filters already normalized, so no division needed
    return integrals

def reconstruct_spectra(photometry, design_matrix):
    """
    Fit linear regression to recover coefficients from photometry.
    Returns estimated coefficients array of shape (n_samples, n_basis).
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix.T, photometry.T)  # transpose to match shape
    coeffs_rec = reg.predict(design_matrix.T).T
    return coeffs_rec

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Wavelength grid (nm)
    wl = np.linspace(400, 800, 401)

    # Create basis functions
    n_basis = 10
    basis_funcs = create_basis_functions(n_basis, wl)

    # Generate synthetic spectra
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis_funcs, rng=rng)

    # Create filter set
    n_filters = 5
    filters = create_filters(n_filters, wl)

    # Compute photometric observations
    photometry = compute_photometry(spectra_true, filters, wl)

    # Build design matrix linking basis to photometry
    design_matrix = build_design_matrix(filters, basis_funcs, wl)

    # Reconstruct spectra coefficients
    coeffs_est = reconstruct_spectra(photometry, design_matrix)

    # Reconstructed spectra
    spectra_rec = coeffs_est @ basis_funcs

    # Evaluate reconstruction error
    mae = np.mean(np.abs(spectra_true - spectra_rec))
    print(f"Mean Absolute Error of reconstructed spectra: {mae:.4f}")

    # Optionally display first spectrum comparison
    idx = 0
    print("\nWavelength (nm) | True Flux | Reconstructed Flux")
    for w, t, r in zip(wl[::20], spectra_true[idx][::20], spectra_rec[idx][::20]):
        print(f"{w:7.1f} | {t:9.4f} | {r:9.4f}")