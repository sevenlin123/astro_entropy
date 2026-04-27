import numpy as np
from sklearn.linear_model import LinearRegression

def gaussian(x, mu, sigma):
    """1‑D Gaussian."""
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def generate_basis(n_basis, wavelength, sigma=20.0):
    """Create a set of Gaussian basis functions."""
    centers = np.linspace(wavelength[0], wavelength[-1], n_basis)
    basis = np.vstack([gaussian(wavelength, c, sigma) for c in centers]).T
    # Normalise each basis function
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return basis

def generate_filters(n_filters, wavelength, width=40.0):
    """Create a set of top‑hat filters."""
    centres = np.linspace(wavelength[0] + width, wavelength[-1] - width, n_filters)
    filters = np.zeros((n_filters, len(wavelength)))
    for i, cen in enumerate(centres):
        mask = np.abs(wavelength - cen) < width/2
        filters[i, mask] = 1.0
    # Normalise each filter
    filt_norm = filters.sum(axis=1, keepdims=True)
    filt_norm[filt_norm == 0] = 1.0
    filters /= filt_norm
    return filters

def synthetic_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs

def photometry_from_spectrum(spectrum, filters):
    """Compute synthetic photometric measurements."""
    return filters @ spectrum

def reconstruct_coeffs(filters, phot, basis):
    """Reconstruct basis coefficients from photometry."""
    # Build the design matrix M = filters @ basis
    M = filters @ basis
    # Solve least‑squares problem M * coeffs ≈ phot
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M, phot)
    return reg.coef_.T

def main():
    # Wavelength grid
    wl = np.linspace(400, 700, 300)          # nm

    # Basis
    n_basis = 15
    basis = generate_basis(n_basis, wl, sigma=15.0)

    # Synthetic coefficients
    rng = np.random.default_rng(seed=42)
    coeff_true = rng.uniform(-1, 1, size=n_basis)

    # True spectrum
    spec_true = synthetic_spectrum(basis, coeff_true)

    # Filters
    n_filters = 7
    filt = generate_filters(n_filters, wl, width=30.0)

    # Observed photometry
    phot_obs = photometry_from_spectrum(spec_true, filt)

    # Reconstruct coefficients
    coeff_rec = reconstruct_coeffs(filt, phot_obs, basis)

    # Reconstructed spectrum
    spec_rec = synthetic_spectrum(basis, coeff_rec)

    # Print summary
    print("True coefficients:", coeff_true[:5], "...")
    print("Recovered coefficients:", coeff_rec[:5], "...")
    print("\nMean absolute error in spectrum:", np.mean(np.abs(spec_true - spec_rec)))

if __name__ == "__main__":
    main()