#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# ---------- Spectral model ----------
def gaussian_spectrum(wavelength, amp, cen, sigma):
    """Single Gaussian line."""
    return amp * np.exp(-0.5 * ((wavelength - cen) / sigma)**2)

def multi_gaussian_spectrum(wavelength, params):
    """Spectrum composed of several Gaussian lines."""
    flux = np.zeros_like(wavelength)
    for amp, cen, sigma in params:
        flux += gaussian_spectrum(wavelength, amp, cen, sigma)
    return flux

def generate_random_params(n_lines, rng=None):
    rng = rng or np.random.default_rng()
    amps   = rng.uniform(0.5, 1.5, size=n_lines)
    cents  = rng.uniform(400, 700, size=n_lines)
    sigmas = rng.uniform(5, 20, size=n_lines)
    return list(zip(amps, cents, sigmas))

# ---------- Filter definitions ----------
def gaussian_filter(wavelength, cen, sigma):
    """Gaussian filter transmission."""
    return np.exp(-0.5 * ((wavelength - cen) / sigma)**2)

def build_filter_matrix(wavelength, filter_centers, filter_sigmas):
    """Construct filter matrix of shape (m, N)."""
    filters = [gaussian_filter(wavelength, c, s) for c, s in zip(filter_centers, filter_sigmas)]
    return np.array(filters)  # shape (m, N)

# ---------- Synthetic data generation ----------
def generate_synthetic_spectra(n_samples, n_lines, wavelength, rng=None):
    rng = rng or np.random.default_rng()
    spectra = []
    params_list = []
    for _ in range(n_samples):
        params = generate_random_params(n_lines, rng=rng)
        spectra.append(multi_gaussian_spectrum(wavelength, params))
        params_list.append(params)
    return np.array(spectra), params_list

def compute_photometry(spectra, filter_matrix, dw=1.0):
    """
    Compute photometric fluxes via matrix multiplication.
    spectra: (n_samples, N)
    filter_matrix: (m, N)
    Returns photometry: (n_samples, m)
    """
    return spectra @ (filter_matrix.T * dw)

# ---------- Reconstruction framework ----------
def train_pca(spectra, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
    pca.fit(spectra)
    return pca.mean_, pca.components_

def reconstruct_from_photometry(photon, mean, components, filter_matrix, dw=1.0):
    """
    Reconstruct spectrum from a single photometric vector.
    photon: (m,)
    """
    # Build matrix mapping coefficients to photometry
    # F*C has shape (m, n_components)
    F_C = filter_matrix @ components.T * dw  # (m, n_components)
    # Solve least squares for coefficients
    coeffs, *_ = np.linalg.lstsq(F_C, photon - filter_matrix @ mean * dw,
                                 rcond=None)
    spectrum_est = mean + components.T @ coeffs
    return spectrum_est

# ---------- Main demonstration ----------
if __name__ == "__main__":
    # Wavelength grid
    N = 300
    wavelength = np.linspace(400, 700, N)  # nm
    dw = wavelength[1] - wavelength[0]

    # Filters: U, B, V (centers 450, 550, 650 nm; sigma 30 nm)
    filter_centers = [450, 550, 650]
    filter_sigmas  = [30, 30, 30]
    filter_mat = build_filter_matrix(wavelength, filter_centers, filter_sigmas)

    # Generate training data
    rng = np.random.default_rng(seed=42)
    n_train = 200
    n_test  = 10
    n_lines = 5
    train_spec, _ = generate_synthetic_spectra(n_train, n_lines, wavelength, rng=rng)
    train_phot = compute_photometry(train_spec, filter_mat, dw)

    # Train PCA
    n_components = 8
    mean_spec, comps = train_pca(train_spec, n_components)

    # Generate test data
    test_spec, test_params = generate_synthetic_spectra(n_test, n_lines, wavelength, rng=rng)
    test_phot = compute_photometry(test_spec, filter_mat, dw)

    # Reconstruction
    recon_specs = []
    for ph in test_phot:
        recon = reconstruct_from_photometry(ph, mean_spec, comps, filter_mat, dw)
        recon_specs.append(recon)
    recon_specs = np.array(recon_specs)

    # Evaluate reconstruction error (root mean square error)
    rmse = np.sqrt(np.mean((test_spec - recon_specs)**2, axis=1))
    print("RMSE per test spectrum:", rmse)
    print("Mean RMSE:", rmse.mean())