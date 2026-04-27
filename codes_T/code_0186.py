#!/usr/bin/env python3
import numpy as np
from scipy.constants import h, c, k
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# Spectral model: Planck function
# ------------------------------------------------------------------
def planck_lambda(wavelength_m, T):
    """Planck function in W m^-2 sr^-1 m^-1."""
    return (2 * h * c**2 / wavelength_m**5) / (
        np.exp(h * c / (wavelength_m * k * T)) - 1
    )

# ------------------------------------------------------------------
# Generate synthetic spectral library
# ------------------------------------------------------------------
def generate_base_spectra(n_basis, wavelengths_m):
    """
    Generate n_basis basis spectra by sampling temperatures.
    Returns array shape (n_basis, len(wavelengths)).
    """
    temps = np.linspace(3000, 10000, n_basis)
    bases = np.array([planck_lambda(wavelengths_m, T) for T in temps])
    return bases

def synthesize_spectrum(coeffs, bases):
    """
    Linear combination of basis spectra.
    coeffs: array shape (n_basis,)
    bases: array shape (n_basis, n_wavelengths)
    Returns spectrum shape (n_wavelengths,)
    """
    return coeffs @ bases

# ------------------------------------------------------------------
# Photometric system
# ------------------------------------------------------------------
def generate_gaussian_filter(center, width, wavelengths_m):
    """
    Gaussian transmission curve.
    center, width in meters.
    """
    return np.exp(-0.5 * ((wavelengths_m - center) / width)**2)

def compute_photometry(spectrum, filters, wavelengths_m):
    """
    Integrate spectrum through each filter.
    spectrum: (n_wavelengths,)
    filters: list of arrays, each shape (n_wavelengths,)
    Returns photometry array shape (n_filters,)
    """
    phot = []
    for filt in filters:
        # Simple trapezoidal integration
        integrand = spectrum * filt
        flux = np.trapz(integrand, wavelengths_m)
        phot.append(flux)
    return np.array(phot)

# ------------------------------------------------------------------
# Training and reconstruction
# ------------------------------------------------------------------
def train_regressor(photometry, spectra, n_components=10):
    """
    Fit a PCA to spectra then regress photometry onto PCA components.
    Returns trained PCA and regression model.
    """
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(spectra)
    reg = LinearRegression()
    reg.fit(photometry, pcs)
    return pca, reg

def reconstruct_spectrum(phot, pca, reg, bases):
    """
    Predict PCA components from photometry and reconstruct spectrum.
    """
    pcs_pred = reg.predict(phot.reshape(1, -1))
    spec_pred = pca.inverse_transform(pcs_pred)
    return spec_pred.flatten()

# ------------------------------------------------------------------
# Main demo
# ------------------------------------------------------------------
def main():
    np.random.seed(42)

    # Define wavelength grid (400-800 nm)
    wl_min, wl_max = 400e-9, 800e-9
    n_wave = 200
    wavelengths = np.linspace(wl_min, wl_max, n_wave)

    # Basis spectra
    n_basis = 15
    bases = generate_base_spectra(n_basis, wavelengths)

    # Filters (7 gaussian filters)
    n_filters = 7
    centers = np.linspace(450e-9, 750e-9, n_filters)
    widths = 30e-9 * np.ones_like(centers)
    filters = [generate_gaussian_filter(c, w, wavelengths) for c, w in zip(centers, widths)]

    # Generate training data
    n_train = 500
    coeffs_train = np.random.rand(n_train, n_basis)
    spectra_train = np.array([synthesize_spectrum(c, bases) for c in coeffs_train])
    photometry_train = np.array([compute_photometry(s, filters, wavelengths) for s in spectra_train])

    # Train model
    pca, reg = train_regressor(photometry_train, spectra_train, n_components=10)

    # Test on new synthetic spectrum
    coeff_true = np.random.rand(n_basis)
    spectrum_true = synthesize_spectrum(coeff_true, bases)
    phot_true = compute_photometry(spectrum_true, filters, wavelengths)

    spectrum_rec = reconstruct_spectrum(phot_true, pca, reg, bases)

    # Evaluate reconstruction error
    mse = np.mean((spectrum_true - spectrum_rec)**2)
    print(f"Reconstruction MSE: {mse:.4e}")

if __name__ == "__main__":
    main()