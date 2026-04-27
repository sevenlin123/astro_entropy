import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Spectral model: linear combination of Gaussian line‐shapes
# ------------------------------------------------------------------
def gaussian(wl, mu, sigma):
    return np.exp(-(wl - mu)**2 / (2 * sigma**2))

def build_basis_functions(wl, n_lines=5):
    """Create a set of Gaussian basis functions."""
    np.random.seed(0)
    mus = np.linspace(wl[0], wl[-1], n_lines)
    sigmas = np.full(n_lines, (wl[-1]-wl[0])/(4*n_lines))
    basis = [gaussian(wl, mu, sigma) for mu, sigma in zip(mus, sigmas)]
    return np.array(basis)  # shape (n_lines, len(wl))

# ------------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------------
def generate_synthetic_spectrum(basis, coeff=None):
    """Generate a spectrum as a weighted sum of basis functions."""
    if coeff is None:
        coeff = np.random.uniform(-1, 1, size=basis.shape[0])
    spec = np.dot(coeff, basis)
    return spec, coeff

# ------------------------------------------------------------------
# Photometric filter definition
# ------------------------------------------------------------------
def make_filter(wl, center, width):
    """Simple top‑hat filter."""
    filt = np.where((wl >= center - width/2) & (wl <= center + width/2), 1.0, 0.0)
    return filt

def generate_filters(wl):
    """Create a few synthetic filters."""
    centers = np.array([450, 550, 650])  # nm
    width   = 100.0                      # nm
    return [make_filter(wl, c, width) for c in centers]

# ------------------------------------------------------------------
# Compute photometry from a spectrum
# ------------------------------------------------------------------
def compute_photometry(spec, filters, wl):
    """Integrate spectrum through each filter."""
    phot = []
    for filt in filters:
        integ = simps(spec * filt, wl)
        phot.append(integ)
    return np.array(phot)

# ------------------------------------------------------------------
# Reconstruct spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_spectrum(phot, filters, basis, wl):
    """
    Estimate coefficients by solving a linear system:
    phot = G * coeff
    where G_ij = integral of basis_i * filter_j
    """
    G = []
    for filt in filters:
        # compute integrals of each basis function through filter
        g_col = [simps(bf * filt, wl) for bf in basis]
        G.append(g_col)
    G = np.array(G).T          # shape (n_lines, n_filters)

    # Ridge regression to avoid ill‑conditioning
    ridge = Ridge(alpha=1e-3, fit_intercept=False)
    ridge.fit(G, phot)
    coeff_est = ridge.coef_
    spec_est = np.dot(coeff_est, basis)
    return spec_est, coeff_est

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (nm)
    wl = np.linspace(400, 800, 1000)

    # Build basis functions
    basis = build_basis_functions(wl, n_lines=7)

    # Generate synthetic spectrum and true coefficients
    true_spec, true_coeff = generate_synthetic_spectrum(basis)

    # Create filters
    filters = generate_filters(wl)

    # Compute photometric measurements
    phot = compute_photometry(true_spec, filters, wl)

    # Reconstruct spectrum from photometry
    rec_spec, rec_coeff = reconstruct_spectrum(phot, filters, basis, wl)

    # Simple output
    print("True coefficients:", true_coeff)
    print("Recovered coefficients:", rec_coeff)
    print("Spectral reconstruction error (RMS):",
          np.sqrt(np.mean((true_spec - rec_spec)**2)))