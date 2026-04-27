import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def make_bases(n_bases, lam):
    """Generate a set of orthogonal basis spectra."""
    np.random.seed(0)
    bases = []
    for i in range(n_bases):
        # Random Gaussian bump
        center = np.random.uniform(lam[0], lam[-1])
        width = np.random.uniform(20, 50)
        amp = np.random.uniform(0.5, 1.5)
        basis = amp * np.exp(-0.5 * ((lam - center) / width)**2)
        bases.append(basis)
    return np.array(bases)

def synth_spectrum(coeffs, bases):
    """Linear combination of basis spectra."""
    return coeffs @ bases

# ----------------------------------------------------------------------
# Photometry
# ----------------------------------------------------------------------
def gaussian_filter_response(lam, center, width):
    """Normalized Gaussian filter response."""
    resp = np.exp(-0.5 * ((lam - center) / width)**2)
    return resp / np.trapz(resp, lam)

def generate_filters(lam):
    """Define a few broadband filters."""
    centers = [500, 600, 700]          # nm
    widths = [40, 50, 60]              # nm
    return [gaussian_filter_response(lam, c, w) for c, w in zip(centers, widths)]

def photometry_from_spectrum(spectrum, lam, filters):
    """Integrate spectrum through each filter."""
    ph = []
    for f in filters:
        ph.append(np.trapz(spectrum * f, lam))
    return np.array(ph)

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def build_design_matrix(bases, lam, filters):
    """Matrix relating coefficients to photometric fluxes."""
    n_filters = len(filters)
    n_bases   = len(bases)
    A = np.zeros((n_filters, n_bases))
    for i, f in enumerate(filters):
        for j, b in enumerate(bases):
            A[i, j] = np.trapz(b * f, lam)
    return A

def reconstruct_coeffs(phot, A):
    """Least‑squares solution for basis coefficients."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A, phot)
    return reg.coef_

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    lam = np.linspace(400, 800, 200)  # nm

    # Basis spectra
    n_bases = 5
    bases   = make_bases(n_bases, lam)

    # True coefficients
    np.random.seed(42)
    true_coeffs = np.random.uniform(0.1, 1.0, n_bases)

    # Generate synthetic spectrum
    true_spec = synth_spectrum(true_coeffs, bases)

    # Filters
    filt = generate_filters(lam)

    # Photometry from synthetic spectrum
    ph_true = photometry_from_spectrum(true_spec, lam, filt)

    # Build design matrix
    A = build_design_matrix(bases, lam, filt)

    # Reconstruct coefficients
    rec_coeffs = reconstruct_coeffs(ph_true, A)

    # Reconstructed spectrum
    rec_spec = synth_spectrum(rec_coeffs, bases)

    # Simple diagnostics
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs  :", rec_coeffs)
    print("Photometry error :", np.linalg.norm(ph_true - photometry_from_spectrum(rec_spec, lam, filt)))
    print("Spectrum L2 error:", np.linalg.norm(true_spec - rec_spec))