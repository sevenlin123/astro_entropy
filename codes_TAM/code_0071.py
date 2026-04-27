#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ---------------------------------------------
# Spectral model utilities
# ---------------------------------------------
def make_basis_functions(wave):
    """Return a list of basis functions evaluated at the given wavelengths."""
    return [np.ones_like(wave), wave, wave ** 2]

def generate_random_coeffs(n_samples, n_basis):
    """Generate random coefficients for each sample."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(size=(n_samples, n_basis))

def generate_spectra(n_samples, wave):
    """Generate synthetic spectra as linear combinations of basis functions."""
    basis = make_basis_functions(wave)
    coeffs = generate_random_coeffs(n_samples, len(basis))
    spectra = np.array([np.dot(c, np.vstack(basis).T) for c in coeffs])
    return spectra, coeffs

# ---------------------------------------------
# Photometry utilities
# ---------------------------------------------
def build_top_hat_filter(wave, center, width):
    """Return a top-hat filter response."""
    response = np.where(np.abs(wave - center) <= width / 2, 1.0, 0.0)
    return response

def compute_photometry(spectra, filters, wave, noise_std=0.02):
    """Compute photometry for each spectrum through the given filters."""
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.empty((n_samples, n_filters))
    for i, filt in enumerate(filters):
        # Mean flux in the band
        integrand = spectra * filt
        flux = simps(integrand, wave) / simps(filt, wave)
        phot[:, i] = flux
    # Add Gaussian noise
    rng = np.random.default_rng(seed=24)
    phot += rng.normal(scale=noise_std, size=phot.shape)
    return phot

# ---------------------------------------------
# Reconstruction utilities
# ---------------------------------------------
def train_reconstruction_model(photometry, spectra):
    """Train a linear regression model mapping photometry to spectra."""
    reg = Ridge(alpha=1.0, fit_intercept=True, random_state=123)
    reg.fit(photometry, spectra)
    return reg

def reconstruct_spectra(model, photometry):
    """Predict spectra from photometry using the trained model."""
    return model.predict(photometry)

# ---------------------------------------------
# Main routine
# ---------------------------------------------
def main():
    # Define wavelength grid (Angstroms)
    wave = np.linspace(4000, 8000, 200)

    # Build filter set (3 top-hat filters)
    filters = [
        build_top_hat_filter(wave, center=5000, width=500),
        build_top_hat_filter(wave, center=6000, width=500),
        build_top_hat_filter(wave, center=7000, width=500),
    ]

    # Generate training data
    n_train = 200
    train_spec, _ = generate_spectra(n_train, wave)
    train_phot = compute_photometry(train_spec, filters, wave)

    # Generate test data
    n_test = 50
    test_spec, _ = generate_spectra(n_test, wave)
    test_phot = compute_photometry(test_spec, filters, wave)

    # Train reconstruction model
    model = train_reconstruction_model(train_phot, train_spec)

    # Predict spectra from test photometry
    pred_spec = reconstruct_spectra(model, test_phot)

    # Evaluate reconstruction quality
    mse = np.mean((pred_spec - test_spec) ** 2)
    print(f"Mean Squared Error on test set: {mse:.4f}")

if __name__ == "__main__":
    main()