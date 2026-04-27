import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----- Spectral Model -----
def basis_functions(wavelength, n_basis):
    """
    Construct simple polynomial basis functions up to degree n_basis-1.
    """
    return np.vstack([wavelength**i for i in range(n_basis)]).T

def generate_synthetic_spectrum(wavelength, coeffs=None, n_basis=5, seed=None):
    """
    Generate a synthetic spectrum as a linear combination of basis functions.
    If coeffs is None, random coefficients are drawn.
    """
    rng = np.random.default_rng(seed)
    if coeffs is None:
        coeffs = rng.uniform(-1, 1, size=n_basis)
    basis = basis_functions(wavelength, n_basis)
    spectrum = basis @ coeffs
    # Normalise spectrum to unit mean
    spectrum /= spectrum.mean()
    return spectrum, coeffs

# ----- Filter Definitions -----
def gaussian_filter(wavelength, center, width):
    """Return a Gaussian transmission curve."""
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def build_filters(wavelength, centers, widths):
    """
    Construct filter transmission curves.
    Returns a list of arrays (same shape as wavelength).
    """
    return [gaussian_filter(wavelength, c, w) for c, w in zip(centers, widths)]

# ----- Photometric Data Generation -----
def photometry_from_spectrum(wavelength, spectrum, filters):
    """
    Integrate the spectrum through each filter.
    Returns an array of fluxes.
    """
    fluxes = []
    for filt in filters:
        # Approximate integral via Simpson's rule
        integrand = spectrum * filt
        flux = simps(integrand, wavelength) / simps(filt, wavelength)
        fluxes.append(flux)
    return np.array(fluxes)

# ----- Reconstruction Framework -----
def build_integrated_basis_matrix(wavelength, filters, n_basis):
    """
    For each filter, compute the integral of each basis function times the
    filter transmission, normalised by the filter area.
    Returns a matrix of shape (n_filters, n_basis).
    """
    basis = basis_functions(wavelength, n_basis)
    mat = []
    for filt in filters:
        integ = simps(basis.T * filt[:, None], wavelength, axis=1)
        norm = simps(filt, wavelength)
        mat.append(integ / norm)
    return np.vstack(mat)

def reconstruct_spectrum_from_photometry(
    fluxes,
    filters,
    wavelength,
    n_basis
):
    """
    Reconstruct the spectrum by solving for coefficients that best reproduce
    the observed photometric fluxes.
    Returns the reconstructed spectrum and recovered coefficients.
    """
    # Build matrix that maps coefficients to photometric fluxes
    A = build_integrated_basis_matrix(wavelength, filters, n_basis)
    # Fit coefficients using linear regression (least-squares)
    reg = LinearRegression(fit_intercept=False).fit(A, fluxes)
    coeffs_rec = reg.coef_
    # Build reconstructed spectrum
    basis = basis_functions(wavelength, n_basis)
    spectrum_rec = basis @ coeffs_rec
    return spectrum_rec, coeffs_rec

# ----- Main Execution -----
if __name__ == "__main__":
    # Wavelength grid (Angstroms)
    wl = np.linspace(3000, 10000, 2000)

    # Basis functions
    n_basis = 6

    # Generate synthetic spectrum
    spec_true, coeff_true = generate_synthetic_spectrum(wl, n_basis=n_basis, seed=42)

    # Define filters (central wavelengths and widths)
    centers = [4000, 5500, 7000, 8500]  # Angstroms
    widths = [300, 300, 300, 300]       # Angstroms

    # Build filter transmission curves
    filters = build_filters(wl, centers, widths)

    # Generate photometric fluxes
    fluxes = photometry_from_spectrum(wl, spec_true, filters)

    # Reconstruct spectrum
    spec_rec, coeff_rec = reconstruct_spectrum_from_photometry(
        fluxes, filters, wl, n_basis
    )

    # Print results
    print("True coefficients:", coeff_true)
    print("Recovered coefficients:", coeff_rec)
    print("\nRelative error in recovered coefficients:",
          np.abs((coeff_rec - coeff_true) / coeff_true))