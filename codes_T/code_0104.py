import numpy as np
from scipy.integrate import simps
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# 1. Basis functions (Legendre polynomials)
def legendre_basis(wavelengths, n_basis):
    """Return basis matrix of shape (len(wavelengths), n_basis)."""
    x = 2 * (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min()) - 1
    basis = np.column_stack([np.polynomial.legendre.Legendre.basis(i)(x) for i in range(n_basis)])
    return basis

# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
def generate_synthetic_spectra(n_spec, wavelengths, n_basis, rng=None):
    """Generate spectra as linear combinations of Legendre basis."""
    rng = np.random.default_rng(rng)
    basis = legendre_basis(wavelengths, n_basis)
    coeffs = rng.normal(size=(n_spec, n_basis))
    spectra = coeffs @ basis.T
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Filter transmission curves (Gaussian bandpasses)
def generate_filters(n_filters, wavelengths, rng=None):
    """Generate Gaussian filter transmission curves."""
    rng = np.random.default_rng(rng)
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_filters)
    widths = rng.uniform((wavelengths.max()-wavelengths.min())/20,
                         (wavelengths.max()-wavelengths.min())/10, size=n_filters)
    filters = np.array([
        np.exp(-0.5 * ((wavelengths - c)/w)**2) for c, w in zip(centers, widths)
    ])
    return filters

# ----------------------------------------------------------------------
# 4. Photometric observations
def photometric_observations(spectra, filters):
    """Compute integrated fluxes through each filter."""
    # Normalize by filter integral to mimic magnitude-like measurement
    filter_norm = simps(filters, axis=1)
    fluxes = spectra @ filters.T
    fluxes /= filter_norm
    return fluxes

# ----------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
def reconstruct_spectra(photometry, filters, wavelengths, n_basis):
    """Recover spectra via least‑squares fitting of basis coefficients."""
    # Build design matrix: filter integrals over basis
    basis = legendre_basis(wavelengths, n_basis)
    A = np.array([simps(basis * f[:, None], axis=0) for f in filters])
    # Solve for coefficients per spectrum
    coeffs_rec = np.linalg.lstsq(A.T, photometry.T, rcond=None)[0].T
    # Reconstruct spectra
    rec_spectra = coeffs_rec @ basis.T
    return rec_spectra, coeffs_rec

# ----------------------------------------------------------------------
# 6. Demo
def main():
    rng = 42
    wavelengths = np.linspace(400, 800, 500)          # nm
    n_spec = 10
    n_filters = 5
    n_basis = 7

    # Generate data
    spectra, coeffs_true = generate_synthetic_spectra(n_spec, wavelengths, n_basis, rng=rng)
    filters = generate_filters(n_filters, wavelengths, rng=rng)
    photometry = photometric_observations(spectra, filters)

    # Reconstruction
    rec_spectra, coeffs_rec = reconstruct_spectra(photometry, filters, wavelengths, n_basis)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(spectra, rec_spectra))
    print(f"Reconstruction RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()