#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model: linear + two Gaussian absorption features
# ----------------------------------------------------------------------
def spectral_model(wl, coeffs):
    """
    Compute synthetic stellar spectrum.

    Parameters
    ----------
    wl : ndarray
        Wavelength grid (nm).
    coeffs : tuple/list
        (continuum level, slope, gauss1 amplitude,
         gauss2 amplitude)

    Returns
    -------
    flux : ndarray
        Flux density on the provided wavelength grid.
    """
    a0, slope, g1_amp, g2_amp = coeffs
    mean_wl = wl.mean()
    range_wl = wl.max() - wl.min()
    cont = a0 + slope * (wl - mean_wl) / range_wl
    g1 = g1_amp * np.exp(-((wl - 500.0) / 20.0) ** 2)
    g2 = g2_amp * np.exp(-((wl - 650.0) / 15.0) ** 2)
    return cont + g1 + g2


# ----------------------------------------------------------------------
# Generate synthetic data
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_stars, wl_grid, rng=np.random.default_rng()):
    """
    Generate `n_stars` synthetic spectra with random coefficients.

    Returns
    -------
    spectra : ndarray (n_stars, len(wl_grid))
    coeffs_list : list of coefficient tuples
    """
    coeffs_list = []
    spectra = []
    for _ in range(n_stars):
        a0 = rng.uniform(0.8, 1.2)
        slope = rng.uniform(-0.3, 0.3)
        g1_amp = rng.uniform(-0.4, -0.05)
        g2_amp = rng.uniform(-0.3, -0.1)
        coeffs = (a0, slope, g1_amp, g2_amp)
        coeffs_list.append(coeffs)
        flux = spectral_model(wl_grid, coeffs)
        # add small Gaussian noise
        flux += rng.normal(scale=0.01, size=flux.shape)
        spectra.append(flux)
    return np.array(spectra), coeffs_list


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------
def gaussian_filter(wl, center, width):
    """Return Gaussian transmission curve."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)


def generate_filters():
    """Define three simple photometric filters."""
    filters = {
        'B': gaussian_filter(wl_grid, 450.0, 30.0),
        'V': gaussian_filter(wl_grid, 550.0, 35.0),
        'I': gaussian_filter(wl_grid, 650.0, 40.0),
    }
    return filters


# ----------------------------------------------------------------------
# Convolution of spectrum with filter
# ----------------------------------------------------------------------
def photometric_flux(spectrum, filt_trans, wl):
    """Compute broadband flux through a filter."""
    return np.trapz(spectrum * filt_trans, wl) / np.trapz(filt_trans, wl)


# ----------------------------------------------------------------------
# Build basis response matrix for filters
# ----------------------------------------------------------------------
def build_response_matrix(filters, wl):
    """For each filter, compute integral of each basis function times filter."""
    # basis functions: continuum offset, slope, gauss1, gauss2
    mean_wl = wl.mean()
    range_wl = wl.max() - wl.min()

    def basis(i, wl):
        if i == 0:
            return np.ones_like(wl)
        elif i == 1:
            return (wl - mean_wl) / range_wl
        elif i == 2:
            return np.exp(-((wl - 500.0) / 20.0) ** 2)
        elif i == 3:
            return np.exp(-((wl - 650.0) / 15.0) ** 2)

    mat = []
    for name, filt in filters.items():
        row = []
        for i in range(4):
            f = basis(i, wl)
            row.append(np.trapz(f * filt, wl) / np.trapz(filt, wl))
        mat.append(row)
    return np.array(mat)  # shape (n_filters, 4)


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra(photometry, response_mat, wl_grid, n_samples=50):
    """
    Reconstruct spectra from photometric fluxes.

    Parameters
    ----------
    photometry : ndarray (n_samples, n_filters)
    response_mat : ndarray (n_filters, n_basis)
    wl_grid : ndarray
        Wavelength grid for reconstruction.
    n_samples : int
        Number of stars (rows in photometry)

    Returns
    -------
    recon_spectra : ndarray (n_samples, len(wl_grid))
    """
    # Solve linear system for each star
    coeffs = np.linalg.lstsq(response_mat, photometry.T, rcond=None)[0].T
    recon_spectra = []
    mean_wl = wl_grid.mean()
    range_wl = wl_grid.max() - wl_grid.min()
    for c in coeffs:
        a0, slope, g1_amp, g2_amp = c
        flux = spectral_model(wl_grid, (a0, slope, g1_amp, g2_amp))
        recon_spectra.append(flux)
    return np.array(recon_spectra)


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid: 400–800 nm, 1000 points
    wl_grid = np.linspace(400.0, 800.0, 1000)

    # Generate synthetic spectra
    n_stars = 30
    spectra, true_coeffs = generate_synthetic_spectra(n_stars, wl_grid)

    # Define filters
    filters = generate_filters()

    # Compute photometric fluxes
    photometry = []
    for spec in spectra:
        flx = [photometric_flux(spec, filt, wl_grid) for filt in filters.values()]
        photometry.append(flx)
    photometry = np.array(photometry)  # shape (n_stars, 3)

    # Build response matrix
    response_mat = build_response_matrix(filters, wl_grid)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(photometry, response_mat, wl_grid, n_samples=n_stars)

    # Simple evaluation: print RMS error for first 5 stars
    rms_errors = np.sqrt(((spectra - recon_spectra) ** 2).mean(axis=1))
    print("RMS errors for first 5 stars:", rms_errors[:5])