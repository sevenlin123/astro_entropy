import numpy as np
from sklearn.linear_model import LinearRegression


def generate_basis(wl, n_basis, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(wl[0], wl[-1], size=n_basis)
    sigmas = rng.uniform(5, 20, size=n_basis)
    basis = [np.exp(-0.5 * ((wl - c) / s) ** 2) for c, s in zip(centers, sigmas)]
    return np.vstack(basis)  # (n_basis, N)


def generate_synthetic_spectrum(basis, coeffs, noise_level=0.01):
    flux = basis.T @ coeffs
    noise = np.random.normal(scale=noise_level, size=flux.shape)
    return flux + noise


def generate_filters(n_filters, wl, seed=1):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(wl[0], wl[-1], size=n_filters)
    sigmas = rng.uniform(10, 30, size=n_filters)
    filters = [np.exp(-0.5 * ((wl - c) / s) ** 2) for c, s in zip(centers, sigmas)]
    return np.vstack(filters)  # (n_filters, N)


def compute_photometry(flux, filters):
    return np.sum(flux * filters, axis=1) / np.sum(filters, axis=1)


def reconstruct_spectrum_from_photometry(filters, photometry, basis):
    proj = np.sum(basis[None, :, :] * filters[:, None, :], axis=2) / np.sum(filters, axis=1)[:, None]
    coeffs = np.linalg.lstsq(proj, photometry, rcond=None)[0]
    recon_flux = basis.T @ coeffs
    return recon_flux, coeffs


def main():
    wl = np.linspace(400, 700, 301)          # wavelength array (nm)
    n_basis = 5
    basis = generate_basis(wl, n_basis, seed=42)

    true_coeffs = np.array([1.5, -0.8, 0.5, 1.2, -0.3])
    flux = generate_synthetic_spectrum(basis, true_coeffs, noise_level=0.02)

    n_filters = 4
    filters = generate_filters(n_filters, wl, seed=24)
    photometry = compute_photometry(flux, filters)

    recon_flux, recon_coeffs = reconstruct_spectrum_from_photometry(filters, photometry, basis)

    print("True coeffs:", true_coeffs)
    print("Reconstructed coeffs:", recon_coeffs)
    print("\nOriginal spectrum (first 10 values):")
    print(flux[:10])
    print("\nReconstructed spectrum (first 10 values):")
    print(recon_flux[:10])


if __name__ == "__main__":
    main()