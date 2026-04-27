import numpy as np
from sklearn.linear_model import LinearRegression


def create_wavelength_grid(nw, lam_min=400, lam_max=800):
    """Uniform wavelength grid in nm."""
    return np.linspace(lam_min, lam_max, nw)


def gaussian_spectrum(wl, amp, cen, sigma):
    """Single Gaussian spectral feature."""
    return amp * np.exp(-0.5 * ((wl - cen) / sigma) ** 2)


def build_basis(wl, nbasis):
    """Build a set of Gaussian basis spectra."""
    rng = np.random.default_rng(42)
    amps = rng.uniform(0.5, 1.5, nbasis)
    cents = rng.uniform(wl.min() + 50, wl.max() - 50, nbasis)
    sigmas = rng.uniform(10, 30, nbasis)
    basis = np.vstack([gaussian_spectrum(wl, a, c, s) for a, c, s in zip(amps, cents, sigmas)])
    return basis


def generate_random_coeffs(nbasis, rng_seed=123):
    """Generate random non‑negative coefficients."""
    rng = np.random.default_rng(rng_seed)
    return rng.dirichlet(np.ones(nbasis))


def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return basis.T @ coeffs


def generate_filters(wl, nfilters):
    """Simple top‑hat filter transmission curves."""
    rng = np.random.default_rng(7)
    trans = np.zeros((nfilters, len(wl)))
    for i in range(nfilters):
        start = rng.uniform(wl.min(), wl.max() - 50)
        width = rng.uniform(20, 60)
        mask = (wl >= start) & (wl <= start + width)
        trans[i, mask] = 1.0
    return trans


def photometry_from_spectrum(spectrum, filters, wl):
    """Integrate spectrum over each filter transmission curve."""
    # assume unit interval for integration
    return (filters @ spectrum) / np.sum(filters, axis=1)


def build_design_matrix(basis, filters, wl):
    """Matrix mapping coefficients to photometric fluxes."""
    nbasis = basis.shape[0]
    nfilters = filters.shape[0]
    design = np.zeros((nfilters, nbasis))
    for j in range(nfilters):
        for k in range(nbasis):
            integrand = filters[j] * basis[k]
            design[j, k] = integrand.sum()
    return design


def reconstruct_coefficients(design_matrix, photometry):
    """Least‑squares recovery of coefficients."""
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design_matrix, photometry)
    return lr.coef_.reshape(-1)


def main():
    # Grid and basis
    wl = create_wavelength_grid(500)
    nbasis = 5
    basis = build_basis(wl, nbasis)

    # Synthetic spectrum
    coeffs_true = generate_random_coeffs(nbasis)
    spec_true = synthesize_spectrum(basis, coeffs_true)

    # Filters and photometry
    nfilters = 7
    filters = generate_filters(wl, nfilters)
    phot_true = photometry_from_spectrum(spec_true, filters, wl)

    # Reconstruction
    design = build_design_matrix(basis, filters, wl)
    coeffs_rec = reconstruct_coefficients(design, phot_true)
    spec_rec = synthesize_spectrum(basis, coeffs_rec)

    # Simple assessment
    print("True coeffs   :", coeffs_true)
    print("Recovered coeffs:", coeffs_rec)
    print("Relative error in spectrum:",
          np.linalg.norm(spec_true - spec_rec) / np.linalg.norm(spec_true))

if __name__ == "__main__":
    main()