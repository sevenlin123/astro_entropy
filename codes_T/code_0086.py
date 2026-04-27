import numpy as np
from scipy import integrate
from sklearn.linear_model import Ridge

# ---------------------- Spectral model ------------------------------------
def gaussian_profile(wave, center, width):
    """Gaussian spectral line."""
    return np.exp(-0.5 * ((wave - center) / width) ** 2)

def generate_synthetic_spectrum(wave_grid, n_lines=5, seed=None):
    """
    Generate a synthetic spectrum as a sum of Gaussian lines with random
    parameters. Returns flux array and a list of parameters.
    """
    rng = np.random.default_rng(seed)
    flux = np.zeros_like(wave_grid)
    params = []
    for _ in range(n_lines):
        center = rng.uniform(wave_grid[0] + 50, wave_grid[-1] - 50)
        width  = rng.uniform(5, 30)
        amplitude = rng.uniform(0.5, 2.0)
        line = amplitude * gaussian_profile(wave_grid, center, width)
        flux += line
        params.append((center, width, amplitude))
    # Add continuum
    continuum = rng.uniform(0.1, 0.5)
    flux += continuum
    return flux, params

# ---------------------- Filter model -------------------------------------
def gaussian_filter(wave, center, width, amplitude=1.0):
    """Transmission curve of a Gaussian filter."""
    return amplitude * np.exp(-0.5 * ((wave - center) / width) ** 2)

def generate_filters(wave_grid, n_filters=5, seed=None):
    """
    Generate a set of synthetic filters (Gaussian transmissions).
    Returns a list of transmission arrays and their parameters.
    """
    rng = np.random.default_rng(seed)
    filters = []
    filt_params = []
    for _ in range(n_filters):
        center = rng.uniform(wave_grid[0] + 100, wave_grid[-1] - 100)
        width  = rng.uniform(20, 80)
        amplitude = rng.uniform(0.8, 1.0)
        filt = gaussian_filter(wave_grid, center, width, amplitude)
        filters.append(filt)
        filt_params.append((center, width, amplitude))
    return filters, filt_params

# ---------------------- Photometry ----------------------------------------
def compute_photometry(spectrum, wave_grid, filters):
    """
    Compute synthetic photometric fluxes by integrating the product of
    the spectrum and each filter transmission over wavelength.
    """
    phot = []
    for filt in filters:
        integrand = spectrum * filt
        flux = integrate.simps(integrand, wave_grid)
        phot.append(flux)
    return np.array(phot)

# ---------------------- Spectrum Reconstruction ---------------------------
def reconstruct_spectrum(photometry, wave_grid, filters, alpha=1.0):
    """
    Reconstruct the spectrum from photometric measurements using
    ridge regression. Builds a matrix A where each column is the
    filter response integrated over wavelength and solves for flux.
    """
    # Build design matrix: each row corresponds to a filter, each column to a wavelength bin
    A = np.array([integrate.simps(f * np.identity(len(wave_grid)), wave_grid)
                  for f in filters]).reshape(len(filters), len(wave_grid))
    # Ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
    ridge.fit(A, photometry)
    reconstructed_flux = ridge.coef_
    return reconstructed_flux

# ---------------------- Main ------------------------------------------------
if __name__ == "__main__":
    # Define wavelength grid (in nm)
    wave_grid = np.linspace(300, 800, 500)

    # Generate synthetic spectrum
    true_spectrum, spec_params = generate_synthetic_spectrum(wave_grid,
                                                             n_lines=7,
                                                             seed=42)

    # Generate synthetic filters
    filters, filt_params = generate_filters(wave_grid,
                                            n_filters=6,
                                            seed=24)

    # Compute photometry
    photometric_fluxes = compute_photometry(true_spectrum, wave_grid, filters)

    # Reconstruct spectrum
    reconstructed_flux = reconstruct_spectrum(photometric_fluxes,
                                               wave_grid,
                                               filters,
                                               alpha=0.5)

    # Print results
    print("True spectrum shape:", true_spectrum.shape)
    print("Photometric fluxes:", photometric_fluxes)
    print("Reconstructed spectrum shape:", reconstructed_flux.shape)
    # Optional: evaluate reconstruction error
    error = np.linalg.norm(true_spectrum - reconstructed_flux) / np.linalg.norm(true_spectrum)
    print(f"Relative L2 reconstruction error: {error:.3f}")