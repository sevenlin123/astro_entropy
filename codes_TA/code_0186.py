import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------
# Spectral model (basis spectra)
# --------------------------------------------------------------------
def get_basis_spectra(n_wave, n_basis=3, seed=42):
    """Return a set of basis spectra defined as Gaussians."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 800, n_wave)   # nm
    basis = []
    centers = rng.uniform(450, 750, size=n_basis)
    widths  = rng.uniform(20, 60, size=n_basis)
    for c, w in zip(centers, widths):
        g = gaussian(n_wave, std=w/(wavelengths[1]-wavelengths[0]))
        g = g / g.max()
        g *= rng.uniform(0.8, 1.2)               # scale factor
        basis.append(g)
    return np.vstack(basis).T   # shape: (n_wave, n_basis)

# --------------------------------------------------------------------
# Synthetic spectra generation
# --------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, coef_bounds=(0.5, 1.5), seed=24):
    """Generate random spectra as linear combos of basis spectra."""
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[1]
    coefs = rng.uniform(coef_bounds[0], coef_bounds[1], size=(n_samples, n_basis))
    spectra = coefs @ basis.T                     # shape: (n_samples, n_wave)
    return spectra, coefs

# --------------------------------------------------------------------
# Filter definitions
# --------------------------------------------------------------------
def define_filters(n_wave, n_filters=3, seed=101):
    """Define simple band‑pass filters (rectangular windows)."""
    rng = np.random.default_rng(seed)
    filters = np.zeros((n_filters, n_wave))
    for i in range(n_filters):
        start = rng.integers(0, n_wave//2)
        stop  = rng.integers(start+10, n_wave)
        filters[i, start:stop] = 1.0
    return filters

# --------------------------------------------------------------------
# Photometric integration
# --------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """Integrate spectra over filter transmission curves."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phots = np.zeros((n_samples, n_filters))
    norm = filters.sum(axis=1, keepdims=True)     # normalise each filter
    for i in range(n_filters):
        phots[:, i] = spectra @ filters[i] / norm[i]
    return phots

# --------------------------------------------------------------------
# Build filter–basis response matrix
# --------------------------------------------------------------------
def build_filter_response_matrix(filters, basis):
    """Compute response matrix R: R[j,i] = ∫ basis_i * filter_j / ∫ filter_j."""
    n_filters, n_wave = filters.shape
    n_basis = basis.shape[1]
    R = np.zeros((n_filters, n_basis))
    norm = filters.sum(axis=1, keepdims=True)
    for j in range(n_filters):
        for i in range(n_basis):
            R[j, i] = (basis[:, i] * filters[j]).sum() / norm[j]
    return R

# --------------------------------------------------------------------
# Reconstruction (coefficients) via pseudoinverse
# --------------------------------------------------------------------
def reconstruct_coefficients(photometry, R):
    """Recover spectrum coefficients from photometry."""
    Rt = R.T
    pinv_Rt = np.linalg.pinv(Rt)
    coeffs_recon = photometry @ pinv_Rt       # shape: (n_samples, n_basis)
    return coeffs_recon

# --------------------------------------------------------------------
# Reconstruct spectra from recovered coefficients
# --------------------------------------------------------------------
def reconstruct_spectra(coeffs, basis):
    """Rebuild full spectra from basis coefficients."""
    return coeffs @ basis.T

# --------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------
def main():
    n_wave = 200                      # number of wavelength points
    n_samples = 5                     # number of synthetic stars

    # 1. Create basis spectra
    basis = get_basis_spectra(n_wave)

    # 2. Generate synthetic spectra & true coefficients
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis)

    # 3. Define filter set
    filters = define_filters(n_wave)

    # 4. Compute photometric data
    photometry = compute_photometry(spectra_true, filters)

    # 5. Build filter–basis response matrix
    R = build_filter_response_matrix(filters, basis)

    # 6. Reconstruct coefficients from photometry
    coeffs_rec = reconstruct_coefficients(photometry, R)

    # 7. Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # ----------------------------------------------------------------
    # Simple diagnostics
    # ----------------------------------------------------------------
    print("True vs Reconstructed Coefficients:")
    print(np.round(np.concatenate([coeffs_true, coeffs_rec]), 3))

    print("\nReconstruction error (L2 norm per spectrum):")
    errors = np.linalg.norm(spectra_true - spectra_rec, axis=1)
    print(errors)

if __name__ == "__main__":
    main()