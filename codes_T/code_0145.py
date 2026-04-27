import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Spectral model – basis of Gaussian components
# ------------------------------------------------------------
def gaussian_basis(n_basis, wavelengths):
    """Return an array of n_basis Gaussian basis spectra."""
    rng = np.random.default_rng(seed=42)
    mus = rng.uniform(400, 800, size=n_basis)          # centers (nm)
    sigmas = rng.uniform(20, 60, size=n_basis)         # widths (nm)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - mus)**2) / sigmas**2)
    return basis


def generate_true_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return basis @ coeffs


# ------------------------------------------------------------
# 2. Synthetic photometry from a spectrum
# ------------------------------------------------------------
def filter_transmission(wavelengths, center, width):
    """Gaussian band‑pass transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center)**2) / width**2)


def generate_filters(wavelengths, centers, widths):
    """Return a list of filter transmission arrays."""
    return [filter_transmission(wavelengths, c, w) for c, w in zip(centers, widths)]


def photometric_flux(spectrum, filters, wavelengths):
    """
    Compute synthetic photometric fluxes by integrating
    the product of spectrum and filter transmission.
    Normalise by the integral of the filter transmission.
    """
    fluxes = []
    for filt in filters:
        num = simps(spectrum * filt, wavelengths)
        den = simps(filt, wavelengths)
        fluxes.append(num / den)
    return np.array(fluxes)


# ------------------------------------------------------------
# 3. Reconstruction framework
# ------------------------------------------------------------
def reconstruct_coeffs_from_photometry(
    photometry, filters, basis, wavelengths, alpha=1e-4
):
    """
    Reconstruct the coefficients of the basis spectra from
    photometric fluxes using Ridge regression.
    """
    # Compute matrix A where each element A[i, j] = ∫ B_j(λ) * F_i(λ) dλ / ∫ F_i dλ
    n_filters = len(filters)
    n_basis = basis.shape[1]
    A = np.empty((n_filters, n_basis))
    for i, filt in enumerate(filters):
        den = simps(filt, wavelengths)
        for j in range(n_basis):
            num = simps(basis[:, j] * filt, wavelengths)
            A[i, j] = num / den

    reg = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    reg.fit(A, photometry)
    return reg.coef_


# ------------------------------------------------------------
# 4. Synthetic example
# ------------------------------------------------------------
def main():
    # Wavelength grid (400–800 nm)
    wav = np.linspace(400, 800, 401)

    # Basis spectra
    n_basis = 10
    basis = gaussian_basis(n_basis, wav)

    # True coefficients (random but positive)
    rng = np.random.default_rng(seed=123)
    true_coeffs = rng.normal(loc=1.0, scale=0.5, size=n_basis)

    # Generate true spectrum
    true_spec = generate_true_spectrum(basis, true_coeffs)

    # Filters: centers at 450, 550, 650 nm; widths 30 nm
    centers = [450, 550, 650]
    widths = [30, 30, 30]
    filters = generate_filters(wav, centers, widths)

    # Synthetic photometry
    photometry = photometric_flux(true_spec, filters, wav)

    # Reconstruction
    recon_coeffs = reconstruct_coeffs_from_photometry(
        photometry, filters, basis, wav, alpha=1e-3
    )
    recon_spec = generate_true_spectrum(basis, recon_coeffs)

    # Output results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", recon_coeffs)
    print("\nTrue spectrum vs reconstructed spectrum plotted over wavelength.\n")

    # Simple difference metric
    diff = np.abs(true_spec - recon_spec).mean()
    print(f"Mean absolute difference between spectra: {diff:.6f}")


if __name__ == "__main__":
    main()