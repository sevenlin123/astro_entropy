#!/usr/bin/env python3
import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model: a set of Gaussian basis functions
# ------------------------------------------------------------------
def gaussian_basis(wavelength, centers, widths):
    """Return an array of shape (n_points, n_bases)."""
    gaussians = []
    for c, w in zip(centers, widths):
        gaussians.append(np.exp(-0.5 * ((wavelength - c) / w) ** 2))
    return np.vstack(gaussians).T

def define_spectral_basis(n_bases=5, wl_min=4000., wl_max=8000.):
    """Create wavelengths and Gaussian basis functions."""
    wavelength = np.linspace(wl_min, wl_max, 500)
    centers = np.linspace(wl_min, wl_max, n_bases)
    widths = np.full(n_bases, (wl_max - wl_min) / (10 * n_bases))
    basis = gaussian_basis(wavelength, centers, widths)
    return wavelength, basis

# ------------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectrum(basis, rng=None):
    """Draw random coefficients and compute synthetic spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    coeffs = rng.uniform(-1, 1, size=basis.shape[1])
    spectrum = basis @ coeffs
    return spectrum, coeffs

# ------------------------------------------------------------------
# Define photometric filters
# ------------------------------------------------------------------
def create_filter(wl_min, wl_max, wavelength):
    """Top‑hat filter between wl_min and wl_max."""
    transmission = np.zeros_like(wavelength)
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    transmission[mask] = 1.0
    return transmission

def define_filters(wavelength):
    """Return list of filter transmissions and names."""
    filters = {
        'u': create_filter(3500, 4000, wavelength),
        'g': create_filter(4000, 5500, wavelength),
        'r': create_filter(5500, 7000, wavelength),
    }
    return filters

# ------------------------------------------------------------------
# Generate photometric fluxes from a spectrum
# ------------------------------------------------------------------
def compute_photometry(spectrum, filters, wavelength):
    """Integrate spectrum over each filter."""
    fluxes = {}
    for name, trans in filters.items():
        # trapezoidal integration
        flux = np.trapz(spectrum * trans, wavelength)
        fluxes[name] = flux
    return fluxes

# ------------------------------------------------------------------
# Reconstruct spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_spectrum(fluxes, basis, filters, wavelength):
    """Solve for basis coefficients that reproduce photometric fluxes."""
    # Build design matrix: each row = integral of each filter over each basis
    n_filters = len(filters)
    n_bases = basis.shape[1]
    A = np.empty((n_filters, n_bases))
    y = np.empty(n_filters)

    for i, (name, trans) in enumerate(filters.items()):
        for j in range(n_bases):
            A[i, j] = np.trapz(basis[:, j] * trans, wavelength)
        y[i] = fluxes[name]

    # Least squares fit
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, y)
    coeffs_rec = lr.coef_
    spectrum_rec = basis @ coeffs_rec
    return spectrum_rec, coeffs_rec

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # Define spectral basis
    wavelength, basis = define_spectral_basis()

    # Generate synthetic spectrum
    true_spectrum, true_coeffs = generate_synthetic_spectrum(basis, rng=rng)

    # Define photometric filters
    filters = define_filters(wavelength)

    # Compute photometric fluxes from synthetic spectrum
    fluxes = compute_photometry(true_spectrum, filters, wavelength)

    # Reconstruct spectrum from photometry
    rec_spectrum, rec_coeffs = reconstruct_spectrum(fluxes, basis, filters, wavelength)

    # Print results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", rec_coeffs)
    print("\nCoefficient error (abs):\n", np.abs(true_coeffs - rec_coeffs))

    # Compare spectra
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wavelength, true_spectrum, label='True spectrum', lw=2)
    plt.plot(wavelength, rec_spectrum, '--', label='Reconstructed spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (arbitrary units)')
    plt.title('Spectrum Reconstruction from Photometry')
    plt.legend()
    plt.tight_layout()
    plt.show()