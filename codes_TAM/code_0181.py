import numpy as np
from sklearn.linear_model import Ridge

def chebyshev_basis(n_wavelengths, n_basis):
    """Generate Chebyshev basis matrix of shape (n_wavelengths, n_basis)."""
    x = np.linspace(-1, 1, n_wavelengths)
    return np.polynomial.chebyshev.chebvander(x, n_basis - 1)

def generate_spectra(n_samples, n_wavelengths, n_basis, rng=None):
    """Generate synthetic spectra as linear combinations of Chebyshev basis."""
    rng = rng or np.random.default_rng()
    coeff = rng.standard_normal((n_samples, n_basis))
    basis = chebyshev_basis(n_wavelengths, n_basis)
    spectra = coeff @ basis.T
    return spectra, coeff

def generate_filters(n_filters, n_wavelengths, rng=None):
    """Generate random filter transmission curves."""
    rng = rng or np.random.default_rng()
    filt = rng.rand(n_filters, n_wavelengths)
    filt /= filt.sum(axis=1, keepdims=True)  # normalize each filter
    return filt

def compute_photometry(spectra, filters):
    """Compute synthetic photometric fluxes by integrating spectra through filters."""
    # spectra: (n_samples, n_wavelengths)
    # filters: (n_filters, n_wavelengths)
    return spectra @ filters.T   # (n_samples, n_filters)

def reconstruct_coefficients(photometry, basis_coeffs, alpha=1.0):
    """Reconstruct spectral basis coefficients from photometry using Ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(photometry, basis_coeffs)
    return ridge.predict(photometry)

def reconstruct_spectra(photometry, filters, n_basis, n_wavelengths, alpha=1.0):
    """Reconstruct spectra from photometry."""
    # Step 1: predict basis coefficients
    basis = chebyshev_basis(n_wavelengths, n_basis)
    # Prepare target basis coefficients: we don't have them here, so we fit on training data.
    # For illustration, we assume photometry corresponds to spectra that we generated.
    # We'll skip actual training and just demonstrate reconstruction pipeline.
    # In practice, you would train ridge on separate data.
    pass  # placeholder

def main():
    rng = np.random.default_rng(42)
    n_samples = 100
    n_wavelengths = 200
    n_basis = 5
    n_filters = 8

    # Generate synthetic spectra
    spectra, coeffs = generate_spectra(n_samples, n_wavelengths, n_basis, rng)

    # Generate filter curves
    filters = generate_filters(n_filters, n_wavelengths, rng)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruct coefficients from photometry
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(photometry, coeffs)
    coeffs_pred = ridge.predict(photometry)

    # Reconstruct spectra
    basis = chebyshev_basis(n_wavelengths, n_basis)
    spectra_rec = coeffs_pred @ basis.T

    print("Spectra shape:", spectra.shape)
    print("Photometry shape:", photometry.shape)
    print("Reconstructed spectra shape:", spectra_rec.shape)

if __name__ == "__main__":
    main()