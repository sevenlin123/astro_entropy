import numpy as np
from scipy.linalg import inv
from numpy.random import uniform, normal

# -----------------------------------------------------------
# Spectral model – sum of Gaussian components
# -----------------------------------------------------------

def gaussian(wl, mu, sigma):
    """One-dimensional Gaussian."""
    return np.exp(-(wl - mu) ** 2 / (2.0 * sigma ** 2))


def synth_spectrum(wl, components):
    """
    Generate a synthetic spectrum.

    Parameters
    ----------
    wl : ndarray
        Wavelength grid.
    components : list of tuples
        Each tuple contains (amplitude, centre, width).

    Returns
    -------
    flux : ndarray
        Spectrum evaluated on `wl`.
    """
    flux = np.zeros_like(wl)
    for amp, mu, sigma in components:
        flux += amp * gaussian(wl, mu, sigma)
    return flux


# -----------------------------------------------------------
# Filter definitions – simple top‑hat filters
# -----------------------------------------------------------

def top_hat_filter(wl, centre, width):
    """Top‑hat filter transmission."""
    return ((np.abs(wl - centre) <= width / 2.0)).astype(float)


def build_filters():
    """Return a list of (name, transmission_array) pairs."""
    centres = [360, 440, 550, 640]   # nm
    width = 50.0                      # nm
    wl = np.linspace(300, 800, 1000)  # common grid for all filters
    return [(f'Filter_{c}', top_hat_filter(wl, c, width)) for c in centres]


# -----------------------------------------------------------
# Photometric integration
# -----------------------------------------------------------

def photometry_from_spectrum(flux, wl, filters):
    """
    Compute synthetic photometric fluxes for each filter.

    Parameters
    ----------
    flux : ndarray
        Input spectrum.
    wl : ndarray
        Wavelength grid.
    filters : list of (name, transmission) tuples

    Returns
    -------
    phot : dict
        Dictionary {filter_name: measured_flux}.
    """
    dlam = wl[1] - wl[0]
    phot = {}
    for name, trans in filters:
        phot[name] = np.sum(flux * trans) * dlam
    return phot


# -----------------------------------------------------------
# Spectrum reconstruction from photometry
# -----------------------------------------------------------

def reconstruct_spectrum(phot, wl, filters, reg=1e-2):
    """
    Reconstruct the spectrum from broadband photometry using
    a regularised least‑squares solution.

    Parameters
    ----------
    phot : dict
        Photometric fluxes {filter_name: value}.
    wl : ndarray
        Wavelength grid used for reconstruction.
    filters : list of (name, transmission) tuples.
    reg : float
        Regularisation parameter (lambda).

    Returns
    -------
    recon_flux : ndarray
        Reconstructed spectrum on `wl`.
    """
    n_filt = len(filters)
    n_wl = len(wl)
    dlam = wl[1] - wl[0]

    # Build design matrix A (filters × wavelengths)
    A = np.zeros((n_filt, n_wl))
    for i, (_, trans) in enumerate(filters):
        A[i, :] = trans * dlam

    # Observation vector
    y = np.array([phot[name] for name, _ in filters])

    # Regularised normal equations
    M = A.T @ A + reg * np.eye(n_wl)
    rhs = A.T @ y
    recon_flux = np.linalg.solve(M, rhs)
    return recon_flux


# -----------------------------------------------------------
# Main routine – generate data and perform reconstruction
# -----------------------------------------------------------

def main():
    # Common wavelength grid
    wl = np.linspace(300, 800, 1000)  # nm

    # Define synthetic spectrum (sum of 3 Gaussians)
    n_components = 3
    comps = []
    rng = np.random.default_rng()
    for _ in range(n_components):
        amp = rng.uniform(1, 5)
        mu = rng.uniform(350, 750)
        sigma = rng.uniform(10, 30)
        comps.append((amp, mu, sigma))
    true_flux = synth_spectrum(wl, comps)

    # Build filter set
    filters = build_filters()

    # Generate synthetic photometry
    phot = photometry_from_spectrum(true_flux, wl, filters)

    # Reconstruct spectrum
    recon_flux = reconstruct_spectrum(phot, wl, filters, reg=1e-1)

    # Evaluate reconstruction
    mse = np.mean((true_flux - recon_flux) ** 2)
    print(f"Mean‑squared error of reconstruction: {mse:.6f}")

    # Optional: print first few values
    for i in range(5):
        print(f"wavelength {wl[i]:>6.1f} nm: true {true_flux[i]:.4f}, recon {recon_flux[i]:.4f}")


if __name__ == "__main__":
    main()