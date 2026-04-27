import numpy as np
from sklearn.linear_model import Ridge


def gaussian(x, mu, sigma, amp):
    """One–dimensional Gaussian."""
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def create_basis(n_basis, wavelengths):
    """Generate a set of basis spectra as random Gaussian combinations."""
    n_wave = len(wavelengths)
    basis = np.zeros((n_basis, n_wave))
    rng = np.random.default_rng()
    for i in range(n_basis):
        n_gauss = rng.integers(1, 4)
        spec = np.zeros_like(wavelengths)
        for _ in range(n_gauss):
            mu = rng.uniform(wavelengths.min(), wavelengths.max())
            sigma = rng.uniform(10, 50)
            amp = rng.uniform(0.5, 2.0)
            spec += gaussian(wavelengths, mu, sigma, amp)
        basis[i] = spec
    # Normalize each basis spectrum
    basis /= np.max(basis, axis=1, keepdims=True)
    return basis


def generate_weights(n_samples, n_basis):
    """Random non‑negative weights for synthetic spectra."""
    rng = np.random.default_rng()
    weights = rng.normal(size=(n_samples, n_basis))
    weights[weights < 0] = 0
    return weights


def generate_spectra(weights, basis):
    """Construct spectra as weighted sums of the basis."""
    return weights @ basis


def create_filters(n_filters, wavelengths):
    """Generate simple bandpasses as Gaussian transmission curves."""
    n_wave = len(wavelengths)
    filters = np.zeros((n_filters, n_wave))
    rng = np.random.default_rng()
    for i in range(n_filters):
        mu = rng.uniform(wavelengths.min(), wavelengths.max())
        sigma = rng.uniform(50, 150)
        filt = gaussian(wavelengths, mu, sigma, 1.0)
        filters[i] = filt / filt.max()  # normalize to unit peak
    return filters


def integrate_over_filter(spectra, filters, wavelengths):
    """
    Compute photometric fluxes by integrating spectra over filter curves.
    spectra : (n_samples, n_wave)
    filters : (n_filters, n_wave)
    returns : (n_samples, n_filters)
    """
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    fluxes = np.zeros((n_samples, n_filters))
    for j in range(n_filters):
        integrand = spectra * filters[j][None, :]  # broadcast over samples
        fluxes[:, j] = np.trapz(integrand, wavelengths, axis=1)
    return fluxes


def compute_basis_photometry(basis, filters, wavelengths):
    """Photometry of the basis spectra (used in reconstruction)."""
    return integrate_over_filter(basis.T, filters, wavelengths).T  # shape (n_filters, n_basis)


def reconstruct_spectra(photometry, basis, filters, wavelengths, alpha=1.0):
    """
    Reconstruct spectra from photometry by solving for basis weights.
    photometry : (n_samples, n_filters)
    returns reconstructed spectra : (n_samples, n_wave)
    """
    M = compute_basis_photometry(basis, filters, wavelengths).T   # (n_basis, n_filters)
    n_samples = photometry.shape[0]
    n_basis = basis.shape[0]
    rec_spectra = np.zeros((n_samples, basis.shape[1]))
    for i in range(n_samples):
        y = photometry[i]  # shape (n_filters,)
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
        ridge.fit(M.T, y)          # M.T : (n_filters, n_basis)
        coeffs = ridge.coef_      # shape (n_basis,)
        rec_spectra[i] = coeffs @ basis
    return rec_spectra


def main():
    # Define wavelength grid
    wav_start, wav_end, dwav = 4000, 8000, 5
    wavelengths = np.arange(wav_start, wav_end + dwav, dwav)

    # Generate basis spectra
    n_basis = 5
    basis = create_basis(n_basis, wavelengths)

    # Generate synthetic dataset
    n_samples = 10
    weights = generate_weights(n_samples, n_basis)
    spectra = generate_spectra(weights, basis)

    # Create filters
    n_filters = 4
    filters = create_filters(n_filters, wavelengths)

    # Compute photometry from synthetic spectra
    photometry = integrate_over_filter(spectra, filters, wavelengths)

    # Reconstruct spectra from photometry
    recon_spectra = reconstruct_spectra(photometry, basis, filters, wavelengths, alpha=0.1)

    # Display comparison for first sample
    idx = 0
    print("Wavelength (Å)\tTrue Flux\tReconstructed Flux")
    for w, t, r in zip(wavelengths, spectra[idx], recon_spectra[idx]):
        print(f"{w:.1f}\t{t:.4f}\t{r:.4f}")


if __name__ == "__main__":
    main()