#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# Spectral model definitions
# ----------------------------------------------------------------------
def generate_wavelength_grid(start=400, stop=700, n_points=301):
    """Linear wavelength grid in nm."""
    return np.linspace(start, stop, n_points)

def generate_basis_spectra(n_basis, n_pixels):
    """
    Create a set of basis spectra.
    Each basis spectrum is a smooth function (Gaussian bumps).
    """
    rng = np.random.default_rng(0)
    wavelengths = np.linspace(0, 1, n_pixels)
    bases = []
    for _ in range(n_basis):
        # Random centers and widths
        centers = rng.uniform(0.2, 0.8, size=rng.integers(1, 4))
        widths = rng.uniform(0.05, 0.15, size=centers.size)
        amp = rng.uniform(0.5, 1.5)
        spectrum = np.zeros_like(wavelengths)
        for c, w in zip(centers, widths):
            spectrum += amp * np.exp(-((wavelengths - c) ** 2) / (2 * w**2))
        # Normalize to unit integral
        spectrum /= np.trapz(spectrum, wavelengths)
        bases.append(spectrum)
    return np.array(bases)  # shape (n_basis, n_pixels)

def generate_filter_transmissions(n_filters, n_pixels):
    """
    Generate synthetic photometric filter transmissions.
    Each filter is a single Gaussian bandpass.
    """
    rng = np.random.default_rng(1)
    wavelengths = np.linspace(0, 1, n_pixels)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(0.2, 0.8)
        width = rng.uniform(0.08, 0.18)
        trans = np.exp(-((wavelengths - center) ** 2) / (2 * width**2))
        # Normalize to unit area for simplicity
        trans /= np.trapz(trans, wavelengths)
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, n_pixels)

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def synthesize_spectrum(coeffs, basis):
    """
    Construct a synthetic spectrum from linear combination of basis spectra.
    coeffs: array of length n_basis
    basis: array of shape (n_basis, n_pixels)
    """
    return np.dot(coeffs, basis)

def compute_photometry(spectrum, filters):
    """
    Integrate spectrum through each filter transmission.
    Returns photometric fluxes for all filters.
    """
    return np.tensordot(filters, spectrum, axes=(1, 0))  # shape (n_filters,)

def create_dataset(n_samples, n_basis, n_filters, basis, filters):
    """
    Generate a dataset of synthetic spectra and their photometry.
    Returns (X_phot, Y_coeffs).
    """
    rng = np.random.default_rng(42)
    # Coefficients drawn from normal distribution, ensure positivity
    coeffs = rng.standard_normal(size=(n_samples, n_basis))
    coeffs = np.abs(coeffs)  # make all positive
    # Optional scaling
    coeffs *= rng.uniform(0.5, 1.5, size=(n_samples, 1))

    spectra = np.dot(coeffs, basis)  # shape (n_samples, n_pixels)
    # Add small Gaussian noise to spectra
    spectra += rng.normal(scale=0.01, size=spectra.shape)

    photometry = np.array([compute_photometry(spec, filters) for spec in spectra])
    return photometry, coeffs, spectra

# ----------------------------------------------------------------------
# Reconstruction pipeline
# ----------------------------------------------------------------------
def train_reconstruction_model(X_train, Y_train):
    """
    Fit a ridge regression model mapping photometry to basis coefficients.
    """
    model = Ridge(alpha=1.0, random_state=0, max_iter=10000)
    model.fit(X_train, Y_train)
    return model

def reconstruct_spectrum(model, photometry, basis):
    """
    Predict coefficients from photometry and reconstruct spectrum.
    """
    coeffs_pred = model.predict(photometry)  # shape (n_samples, n_basis)
    return np.dot(coeffs_pred, basis)       # shape (n_samples, n_pixels)

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    # Parameters
    n_pixels = 301          # number of wavelength points
    n_basis = 6             # number of basis spectra
    n_filters = 10          # number of photometric filters
    n_train = 1200          # training samples
    n_test = 300            # test samples

    # Generate model components
    wavelengths = generate_wavelength_grid()
    basis = generate_basis_spectra(n_basis, n_pixels)
    filters = generate_filter_transmissions(n_filters, n_pixels)

    # Build datasets
    X_train, coeffs_train, spectra_train = create_dataset(
        n_train, n_basis, n_filters, basis, filters
    )
    X_test, coeffs_test, spectra_test = create_dataset(
        n_test, n_basis, n_filters, basis, filters
    )

    # Train regression model
    model = train_reconstruction_model(X_train, coeffs_train)

    # Reconstruct test spectra
    spectra_rec = reconstruct_spectrum(model, X_test, basis)

    # Evaluate reconstruction error
    mse_per_pixel = mean_squared_error(spectra_test, spectra_rec, multioutput='raw_values')
    print(f"Mean squared error per pixel: {mse_per_pixel.mean():.6f}")

    # Example output for first test sample
    idx = 0
    print("\nTrue vs. reconstructed spectrum for first test sample:")
    print("Wavelength (nm)\tTrue\tReconstructed")
    for w, t, r in zip(wavelengths, spectra_test[idx], spectra_rec[idx]):
        print(f"{w:.1f}\t{t:.4f}\t{r:.4f}")

if __name__ == "__main__":
    main()