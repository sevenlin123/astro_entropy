import numpy as np
from sklearn.linear_model import LinearRegression

def spectral_grid(n_points=500, wl_min=4000, wl_max=7000):
    """Wavelength grid in Angstroms."""
    return np.linspace(wl_min, wl_max, n_points)

def gaussian_line(wl, amp, cen, width):
    """Single Gaussian absorption/emission line."""
    return amp * np.exp(-((wl - cen) ** 2) / (2 * width ** 2))

def generate_basis(n_basis=5, wl_grid=None):
    """Generate a set of Gaussian basis spectra."""
    if wl_grid is None:
        wl_grid = spectral_grid()
    basis = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_basis):
        amp = rng.uniform(0.5, 1.5)
        cen = rng.uniform(wl_grid[0], wl_grid[-1])
        width = rng.uniform(10, 50)
        basis.append(gaussian_line(wl_grid, amp, cen, width))
    return np.vstack(basis)  # shape (n_basis, n_wl)

def generate_synthetic_spectrum(basis, weights=None):
    """Linear combination of basis spectra to create a synthetic spectrum."""
    if weights is None:
        rng = np.random.default_rng(seed=24)
        weights = rng.uniform(0.5, 1.5, size=basis.shape[0])
    spectrum = np.dot(weights, basis)
    return spectrum, weights

def create_filters(n_filters=4, wl_grid=None):
    """Simple top‑hat filters."""
    if wl_grid is None:
        wl_grid = spectral_grid()
    rng = np.random.default_rng(seed=99)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wl_grid[0], wl_grid[-1])
        width = rng.uniform(100, 300)
        f = np.zeros_like(wl_grid)
        mask = np.abs(wl_grid - center) <= width / 2
        f[mask] = 1.0
        filters.append(f)
    return np.vstack(filters)  # shape (n_filters, n_wl)

def compute_photometry(spectrum, filters, wl_grid):
    """Integrate spectrum through each filter."""
    phot = []
    for f in filters:
        # simple weighted mean: ∫ Sλ Tλ dλ / ∫ Tλ dλ
        numerator = np.trapz(spectrum * f, wl_grid)
        denominator = np.trapz(f, wl_grid)
        phot.append(numerator / denominator if denominator > 0 else 0.0)
    return np.array(phot)

def reconstruct_weights(basis, filters, photometry):
    """
    Fit linear model: basis integrals × weights ≈ photometry.
    Returns recovered weights.
    """
    # compute matrix A where A_ij = ∫ basis_j * filter_i dλ
    n_filters, n_wl = filters.shape
    n_basis = basis.shape[0]
    A = np.empty((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            A[i, j] = np.trapz(basis[j] * filters[i], axis=0)
    # linear regression without regularization
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A, photometry)
    return reg.coef_

def main():
    # grid and basis
    wl_grid = spectral_grid()
    basis = generate_basis(n_basis=5, wl_grid=wl_grid)

    # synthetic spectrum
    true_spectrum, true_weights = generate_synthetic_spectrum(basis)

    # filters and photometry
    filters = create_filters(n_filters=4, wl_grid=wl_grid)
    photometry = compute_photometry(true_spectrum, filters, wl_grid)

    # reconstruction
    recovered_weights = reconstruct_weights(basis, filters, photometry)
    recovered_spectrum = np.dot(recovered_weights, basis)

    # output results
    print("True weights:", true_weights)
    print("Recovered weights:", recovered_weights)
    print("\nDifference in spectrum (L2 norm):", np.linalg.norm(true_spectrum - recovered_spectrum))

if __name__ == "__main__":
    main()