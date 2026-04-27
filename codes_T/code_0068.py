import numpy as np
from sklearn.linear_model import LinearRegression


def wavelength_grid(start=3000, end=10000, npts=1000):
    """Create a wavelength grid (in nm)."""
    return np.linspace(start, end, npts)


def gaussian_basis(nbasis, wl, cov=200):
    """
    Generate a set of Gaussian basis functions.
    :param nbasis: number of basis functions
    :param wl: wavelength grid
    :param cov: standard deviation of Gaussians
    :return: basis matrix of shape (len(wl), nbasis)
    """
    centers = np.linspace(wl.min(), wl.max(), nbasis)
    basis = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) ** 2) / cov**2)
    return basis


def random_coeffs(nbasis, rng=None):
    """
    Sample random coefficients uniformly from [-1, 1].
    """
    rng = np.random.default_rng(rng)
    return rng.uniform(-1, 1, size=nbasis)


def synthetic_spectrum(basis, coeffs):
    """
    Construct a synthetic spectrum from basis functions and coefficients.
    """
    return basis @ coeffs


def top_hat_filters(nfilters, wl, widths=(500, 700)):
    """
    Create top‑hat filter transmission curves.
    :param nfilters: number of filters
    :param wl: wavelength grid
    :param widths: tuple (min_width, max_width) in nm
    :return: filter matrix shape (nfilters, len(wl))
    """
    centers = np.linspace(wl.min() + 200, wl.max() - 200, nfilters)
    filter_mat = np.zeros((nfilters, len(wl)))
    for i, cen in enumerate(centers):
        w = np.random.uniform(*widths)
        mask = (wl >= cen - w / 2) & (wl <= cen + w / 2)
        filter_mat[i, mask] = 1.0
    return filter_mat


def photometry_from_spectrum(spectrum, filters, wl):
    """
    Compute synthetic photometric fluxes for a spectrum.
    """
    integrals = np.trapz(spectrum[:, None] * filters, wl, axis=1)
    norms = np.trapz(filters, wl, axis=1)
    return integrals / norms


def build_design_matrix(basis, filters, wl):
    """
    Build the linear design matrix relating coefficients to photometric fluxes.
    """
    nfilters, npoints = filters.shape
    nbasis = basis.shape[1]
    M = np.empty((nfilters, nbasis))
    norms = np.trapz(filters, wl, axis=1)
    for j in range(nfilters):
        for i in range(nbasis):
            M[j, i] = np.trapz(basis[:, i] * filters[j], wl) / norms[j]
    return M


def reconstruct_spectrum(design_mat, phot, basis, wl):
    """
    Reconstruct the spectrum from photometry using linear regression.
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design_mat, phot)
    coeffs_est = lr.coef_
    recon_spec = basis @ coeffs_est
    return recon_spec, coeffs_est


def main():
    # Setup wavelength grid
    wl = wavelength_grid()

    # Define basis and generate random coefficients
    nbasis = 20
    basis = gaussian_basis(nbasis, wl, cov=300)
    true_coeffs = random_coeffs(nbasis, rng=42)

    # Generate synthetic spectrum
    spec = synthetic_spectrum(basis, true_coeffs)

    # Create filters and compute photometry
    nfilters = 10
    filters = top_hat_filters(nfilters, wl, widths=(400, 800))
    phot = photometry_from_spectrum(spec, filters, wl)

    # Reconstruction
    design_mat = build_design_matrix(basis, filters, wl)
    recon_spec, est_coeffs = reconstruct_spectrum(design_mat, phot, basis, wl)

    # Evaluate reconstruction quality
    rms_error = np.sqrt(np.mean((spec - recon_spec) ** 2))
    print(f"RMS reconstruction error: {rms_error:.4f}")


if __name__ == "__main__":
    main()