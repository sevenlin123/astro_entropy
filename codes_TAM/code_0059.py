import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model and basis functions
# ----------------------------------------------------------------------
def gaussian_basis(center, sigma, wavelengths):
    """Return a Gaussian basis spectrum."""
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def create_basis(n_basis, wavelengths):
    """Generate n_basis Gaussian basis spectra with random centers and widths."""
    rng = np.random.default_rng(42)
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_basis)
    sigmas  = rng.uniform(20, 50, size=n_basis)          # arbitrary width range
    return np.vstack([gaussian_basis(c, s, wavelengths) for c, s in zip(centers, sigmas)])

def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return coeffs @ basis

# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, n_basis, noise_std=0.01):
    """Generate n_samples synthetic spectra with random coefficients."""
    rng = np.random.default_rng(1234)
    coeffs = rng.normal(scale=1.0, size=(n_samples, n_basis))
    basis  = create_basis(n_basis, wavelengths)
    spectra = np.array([synthesize_spectrum(c, basis) for c in coeffs])
    spectra += rng.normal(scale=noise_std, size=spectra.shape)   # add noise
    return spectra, coeffs, basis

# ----------------------------------------------------------------------
# Photometric simulation
# ----------------------------------------------------------------------
def filter_response(center, width, wavelengths):
    """Simple Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra through given filter responses."""
    photometry = []
    for filt in filters:
        resp = filter_response(filt["center"], filt["width"], wavelengths)
        flux = simps(spectra * resp, wavelengths, axis=1)
        photometry.append(flux)
    return np.column_stack(photometry)

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra(photometry, filters, wavelengths, basis):
    """Recover spectra by fitting basis coefficients to photometry."""
    # Build response matrix M (filters x basis)
    M = np.empty((len(filters), basis.shape[0]))
    for i, filt in enumerate(filters):
        resp = filter_response(filt["center"], filt["width"], wavelengths)
        for j in range(basis.shape[0]):
            M[i, j] = simps(basis[j] * resp, wavelengths)
    # Fit coefficients for each spectrum using Ridge regression
    ridge = Ridge(alpha=1e-3, fit_intercept=False, solver='svd')
    ridge.fit(M.T, photometry.T)  # transposed to match sklearn shape
    coeffs_rec = ridge.coef_.T
    # Reconstruct spectra
    reconstructed = coeffs_rec @ basis
    return reconstructed, coeffs_rec

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def main():
    wavelengths = np.linspace(400, 800, 401)  # 400-800 nm
    n_basis = 5
    n_samples = 10

    # Generate synthetic spectra
    spectra_true, coeffs_true, basis = generate_synthetic_spectra(n_samples, wavelengths, n_basis)

    # Define photometric filters
    filters = [
        {"center": 450, "width": 20},
        {"center": 550, "width": 30},
        {"center": 650, "width": 25},
    ]

    # Compute photometry
    photometry = compute_photometry(spectra_true, filters, wavelengths)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filters, wavelengths, basis)

    # Print comparison of first spectrum
    print("True spectrum first sample (first 5 values):", spectra_true[0][:5])
    print("Reconstructed spectrum first sample (first 5 values):", spectra_rec[0][:5])

if __name__ == "__main__":
    main()