import numpy as np
from sklearn.linear_model import Ridge


def wavelength_grid(start=400, stop=800, step=5):
    """Create a wavelength array in nm."""
    return np.arange(start, stop + step, step)


def basis_functions(wave):
    """Return a list of basis spectra (Gaussian peaks)."""
    g1 = np.exp(-0.5 * ((wave - 450) / 20) ** 2)
    g2 = np.exp(-0.5 * ((wave - 550) / 30) ** 2)
    g3 = np.exp(-0.5 * ((wave - 650) / 25) ** 2)
    return [g1, g2, g3]


def generate_synthetic_spectra(n_samples, wave, n_basis=3):
    """Generate synthetic spectra as random linear combos of basis functions."""
    bases = basis_functions(wave)[:n_basis]
    coeffs = np.random.uniform(0.5, 1.5, size=(n_samples, n_basis))
    spectra = np.dot(coeffs, np.array(bases))
    # Add small Gaussian noise
    spectra += np.random.normal(scale=0.01, size=spectra.shape)
    return spectra, coeffs


def gaussian_filter(wave, center, width):
    """Return a Gaussian bandpass transmission curve."""
    return np.exp(-0.5 * ((wave - center) / width) ** 2)


def make_filters(num_filters, wave):
    """Generate a list of random Gaussian filters."""
    rng = np.random.default_rng(seed=42)
    centers = rng.uniform(420, 780, size=num_filters)
    widths = rng.uniform(20, 50, size=num_filters)
    filters = [gaussian_filter(wave, c, w) for c, w in zip(centers, widths)]
    return filters


def photometry_from_spectra(spectra, filters, wave):
    """Integrate spectra through each filter."""
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.empty((n_samples, n_filters))
    for i in range(n_filters):
        trans = filters[i]
        # Trapezoidal integration over wavelength
        phot[:, i] = np.trapz(spectra * trans, wave, axis=1)
    return phot


def forward_matrix(filters, bases, wave):
    """Compute integral of each basis through each filter."""
    n_filters = len(filters)
    n_bases = len(bases)
    A = np.empty((n_filters, n_bases))
    for i, f in enumerate(filters):
        for j, b in enumerate(bases):
            A[i, j] = np.trapz(b * f, wave)
    return A


def reconstruct_spectra(phot, filters, wave, n_basis=3, alpha=0.1):
    """Reconstruct spectra from photometry using ridge regression."""
    bases = basis_functions(wave)[:n_basis]
    A = forward_matrix(filters, bases, wave)
    # Solve for coefficients per sample
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A.T, phot.T)          # Solve A^T beta = phot^T
    coeffs_hat = ridge.predict(np.eye(A.shape[0]))  # Equivalent to (A^T A + αI)^(-1) A^T
    coeffs_hat = ridge.coef_.T      # shape (n_samples, n_basis)
    # Reconstruct spectra
    recon = np.dot(coeffs_hat, np.array(bases))
    return recon, coeffs_hat


if __name__ == "__main__":
    np.random.seed(0)

    # 1. Define wavelengths and basis spectra
    wav = wavelength_grid()
    n_samples = 10

    # 2. Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, wav)

    # 3. Generate filters
    n_filters = 5
    filters = make_filters(n_filters, wav)

    # 4. Compute photometric data
    phot = photometry_from_spectra(spectra_true, filters, wav)

    # 5. Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(phot, filters, wav)

    # 6. Evaluate reconstruction
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error of reconstruction: {mse:.6f}")