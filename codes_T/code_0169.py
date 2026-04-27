import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge


def generate_wavelengths(start=400., stop=700., step=1.):
    """Create a regular wavelength grid in nm."""
    return np.arange(start, stop + step, step)


def gaussian_basis(n_basis, wavelengths):
    """
    Create a set of Gaussian basis spectra.
    Each basis is a Gaussian centered at evenly spaced positions.
    """
    centers = np.linspace(wavelengths.min() + 50,
                          wavelengths.max() - 50, n_basis)
    widths = 30.0 * np.ones(n_basis)
    basis = np.array(
        [np.exp(-0.5 * ((wavelengths - c) / w)**2) for c, w in zip(centers, widths)]
    )
    return basis  # shape (n_basis, n_wavelength)


def generate_filters(n_filters, wavelengths):
    """
    Create simple Gaussian bandpass filters.
    Centers are evenly spaced across the wavelength range.
    """
    centers = np.linspace(wavelengths.min() + 25,
                          wavelengths.max() - 25, n_filters)
    widths = 30.0 * np.ones(n_filters)
    filters = np.array(
        [np.exp(-0.5 * ((wavelengths - c) / w)**2) for c, w in zip(centers, widths)]
    )
    return filters  # shape (n_filters, n_wavelength)


def synth_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return coeffs @ basis  # shape (n_wavelength,)


def photometric_flux(spectrum, filters, wavelengths):
    """
    Compute photometric fluxes by integrating spectrum times filter response.
    """
    fluxes = []
    for f in filters:
        integrand = spectrum * f
        flux = trapz(integrand, wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)  # shape (n_filters,)


def reconstruct_coeffs_from_photometry(fluxes, filters, wavelengths, basis, alpha=1e-3):
    """
    Reconstruct the coefficients using ridge regression.
    The design matrix is the integral of each basis spectrum through each filter.
    """
    # Build design matrix A where A_ij = ∫ (basis_j * filter_i) dλ
    n_filters, n_wavelength = filters.shape[0], wavelengths.size
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            integrand = basis[j] * filters[i]
            A[i, j] = trapz(integrand, wavelengths)

    # Solve for coefficients
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(A, fluxes)
    return reg.coef_


def main():
    # Parameters
    n_basis = 3
    n_filters = 5
    wavelengths = generate_wavelengths()
    basis = gaussian_basis(n_basis, wavelengths)
    filters = generate_filters(n_filters, wavelengths)

    # True coefficients
    true_coeffs = np.array([1.5, -0.3, 0.8])

    # Generate synthetic spectrum
    true_spectrum = synth_spectrum(true_coeffs, basis)

    # Generate synthetic photometry
    fluxes = photometric_flux(true_spectrum, filters, wavelengths)

    # Reconstruct coefficients from photometry
    recon_coeffs = reconstruct_coeffs_from_photometry(fluxes, filters,
                                                       wavelengths, basis)

    # Reconstruct spectrum
    recon_spectrum = synth_spectrum(recon_coeffs, basis)

    # Print results
    print("True coefficients   :", true_coeffs)
    print("Recovered coefficients :", recon_coeffs)
    print("Spectrum reconstruction error (RMS):",
          np.sqrt(np.mean((true_spectrum - recon_spectrum)**2)))


if __name__ == "__main__":
    main()