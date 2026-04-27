import numpy as np
from scipy.linalg import lstsq

# ------------------------------------------------------------------
# Spectral model: a set of basis spectra (gaussian peaks)
# ------------------------------------------------------------------
def generate_basis(num_basis, num_wavelengths, seed=0):
    """
    Generate a set of basis spectra as Gaussian functions.

    Parameters
    ----------
    num_basis : int
        Number of basis spectra to generate.
    num_wavelengths : int
        Length of each spectrum (number of wavelength points).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    basis : ndarray, shape (num_wavelengths, num_basis)
        Basis spectra matrix.
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 800, num_wavelengths)  # nm
    basis = np.zeros((num_wavelengths, num_basis))
    for i in range(num_basis):
        mu = rng.uniform(400, 800)
        sigma = rng.uniform(20, 60)
        amplitude = rng.uniform(0.5, 1.5)
        basis[:, i] = amplitude * np.exp(-0.5 * ((wavelengths - mu)/sigma)**2)
    return basis


# ------------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------------
def generate_synthetic_spectra(num_samples, basis, seed=42):
    """
    Create synthetic spectra as random linear combinations of basis spectra.

    Parameters
    ----------
    num_samples : int
        Number of synthetic spectra to generate.
    basis : ndarray, shape (n_wavelengths, n_basis)
        Basis spectra.
    seed : int
        Random seed.

    Returns
    -------
    spectra : ndarray, shape (n_wavelengths, num_samples)
        Generated spectra.
    coeffs : ndarray, shape (n_basis, num_samples)
        Coefficients used for reconstruction.
    """
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[1]
    coeffs = rng.normal(size=(n_basis, num_samples))
    spectra = basis @ coeffs
    return spectra, coeffs


# ------------------------------------------------------------------
# Photometric data generation
# ------------------------------------------------------------------
def generate_photometry(spectra, response_matrix, noise_std=0.01, seed=7):
    """
    Simulate photometric observations by integrating spectra through filter responses.

    Parameters
    ----------
    spectra : ndarray, shape (n_wavelengths, n_samples)
        Input spectra.
    response_matrix : ndarray, shape (n_filters, n_wavelengths)
        Filter transmission curves.
    noise_std : float
        Standard deviation of Gaussian noise added to photometry.
    seed : int
        Random seed.

    Returns
    -------
    photometry : ndarray, shape (n_filters, n_samples)
        Simulated photometric fluxes.
    """
    rng = np.random.default_rng(seed)
    noiseless = response_matrix @ spectra
    noise = rng.normal(scale=noise_std, size=noiseless.shape)
    return noiseless + noise


# ------------------------------------------------------------------
# Spectrum reconstruction from photometry
# ------------------------------------------------------------------
def reconstruct_spectra_from_photometry(
        photometry, basis, response_matrix, regularization=0.0):
    """
    Reconstruct spectra as linear combinations of basis spectra given photometry.

    Parameters
    ----------
    photometry : ndarray, shape (n_filters, n_samples)
        Observed photometric fluxes.
    basis : ndarray, shape (n_wavelengths, n_basis)
        Basis spectra.
    response_matrix : ndarray, shape (n_filters, n_wavelengths)
        Filter response matrix.
    regularization : float
        Tikhonov regularization parameter (lambda). 0.0 disables.

    Returns
    -------
    recon_spectra : ndarray, shape (n_wavelengths, n_samples)
        Reconstructed spectra.
    coeffs : ndarray, shape (n_basis, n_samples)
        Coefficients of basis spectra.
    """
    n_filters, n_samples = photometry.shape
    n_basis = basis.shape[1]

    # Effective forward matrix mapping coefficients to photometry:
    A = response_matrix @ basis            # shape (n_filters, n_basis)

    # Solve least squares for each sample:
    coeffs = np.empty((n_basis, n_samples))
    for i in range(n_samples):
        # Regularized LS solution: (A^T A + λI) x = A^T y
        if regularization > 0.0:
            AtA = A.T @ A + regularization * np.eye(n_basis)
            Atb = A.T @ photometry[:, i]
            coeffs[:, i] = np.linalg.solve(AtA, Atb)
        else:
            coeffs[:, i], *_ = lstsq(A, photometry[:, i])

    recon_spectra = basis @ coeffs
    return recon_spectra, coeffs


# ------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------
def main():
    # Settings
    n_wavelengths = 100
    n_basis = 5
    n_filters = 10
    n_samples = 20
    noise_std = 0.02

    # Generate basis spectra
    basis = generate_basis(n_basis, n_wavelengths, seed=0)

    # Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis, seed=42)

    # Generate random filter responses
    rng = np.random.default_rng(99)
    response_matrix = rng.normal(size=(n_filters, n_wavelengths))
    # Normalize each filter
    response_matrix /= np.linalg.norm(response_matrix, axis=1, keepdims=True)

    # Generate photometric observations
    photometry = generate_photometry(spectra_true, response_matrix,
                                     noise_std=noise_std, seed=7)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra_from_photometry(
        photometry, basis, response_matrix, regularization=1e-4)

    # Evaluate reconstruction error
    mse = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {mse:.6f}")

    # Optional: compare true vs recovered coefficients for one sample
    sample = 0
    print("\nTrue coefficients (sample 0):", coeffs_true[:, sample])
    print("Recovered coefficients (sample 0):", coeffs_rec[:, sample])


if __name__ == "__main__":
    main()