import numpy as np
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def build_basis_functions(wl, n_basis=5, seed=42):
    """
    Build an array of basis spectra.
    Each basis is a Gaussian with random center and width.
    Returns:
        basis_matrix : shape (n_basis, len(wl))
        coeffs_true : random true coefficients used for synthesis
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(450, 750, size=n_basis)
    sigmas = rng.uniform(20, 60, size=n_basis)
    basis = np.exp(-((wl[:, None] - centers)**2) / (2 * sigmas**2))
    coeffs_true = rng.normal(scale=1.0, size=n_basis)
    return basis, coeffs_true

def generate_synthetic_spectra(n_spectra, wl, basis, rng=None):
    """
    Generate synthetic spectra by random linear combinations of basis functions.
    Returns:
        spectra : (n_spectra, len(wl))
        coeffs   : (n_spectra, n_basis)
    """
    if rng is None:
        rng = np.random.default_rng()
    n_basis = basis.shape[0]
    coeffs = rng.normal(scale=1.0, size=(n_spectra, n_basis))
    spectra = coeffs @ basis  # shape (n_spectra, len(wl))
    # add small Gaussian noise
    noise = rng.normal(scale=0.02, size=spectra.shape)
    return spectra + noise, coeffs

# ---------- Photometric bandpasses ----------
def top_hat_filter(wl, center, width):
    """Return a top‑hat transmission curve."""
    return np.where((wl >= center - width/2) & (wl <= center + width/2), 1.0, 0.0)

def build_bandpasses(wl, n_filters=5, seed=24):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(500, 700, size=n_filters)
    widths = rng.uniform(40, 80, size=n_filters)
    filters = [top_hat_filter(wl, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)          # shape (n_filters, len(wl))

def compute_photometry(spectra, filters, wl):
    """
    Integrate spectra over each filter to obtain synthetic photometry.
    """
    n_spectra = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_spectra, n_filters))
    for i in range(n_filters):
        trans = filters[i]
        phot[:, i] = trapezoid(spectra.T * trans, wl, axis=0)
    return phot

# ---------- Reconstruction ----------
def construct_design_matrix(filters, basis, wl):
    """
    Compute the matrix that maps basis coefficients to photometric fluxes.
    Each entry A_ij = ∫ B_j(λ) * F_i(λ) dλ
    """
    n_filters = filters.shape[0]
    n_basis   = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            A[i, j] = trapezoid(basis[j] * filters[i], wl)
    return A

def reconstruct_coefficients(phot, A):
    """
    Solve least‑squares problem A * coeffs = phot for each spectrum.
    Returns coefficient matrix of shape (n_spectra, n_basis).
    """
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(A, phot.T)
    coeffs_rec = reg.coef_.T
    return coeffs_rec

def reconstruct_spectrum(coeffs, basis):
    """
    Reconstruct spectrum from recovered coefficients.
    """
    return coeffs @ basis

# ---------- Main pipeline ----------
def main():
    rng = np.random.default_rng(12345)

    # Wavelength grid
    wl = np.linspace(400, 800, 400)          # 400–800 nm

    # Basis functions
    basis, coeffs_true = build_basis_functions(wl, n_basis=5, seed=101)

    # Synthetic spectra
    n_spectra = 30
    spectra, coeffs_orig = generate_synthetic_spectra(n_spectra, wl, basis, rng=rng)

    # Bandpasses
    filters = build_bandpasses(wl, n_filters=6, seed=202)

    # Photometric observations
    phot = compute_photometry(spectra, filters, wl)

    # Design matrix
    A = construct_design_matrix(filters, basis, wl)

    # Coefficient reconstruction
    coeffs_rec = reconstruct_coefficients(phot, A)

    # Spectrum reconstruction
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)

    # Simple sanity check: mean absolute error between original and reconstructed spectra
    mae = np.mean(np.abs(spectra - spectra_rec), axis=1)
    print("Mean absolute reconstruction error per spectrum:",
          np.round(mae, 4))

if __name__ == "__main__":
    main()