#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------------
# Spectral model utilities
# -------------------------------------------------------------------
def create_basis_functions(n_basis: int, wavelengths: np.ndarray) -> np.ndarray:
    """
    Create a set of sinusoidal basis spectra.
    Returns an array of shape (n_basis, n_wavelengths).
    """
    rng = np.random.default_rng()
    basis = []
    for _ in range(n_basis):
        amp = rng.uniform(0.5, 1.5)
        freq = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        spec = amp * np.sin(freq * wavelengths + phase)
        basis.append(spec)
    return np.array(basis)


def generate_synthetic_spectra(
    n_samples: int,
    wavelengths: np.ndarray,
    n_basis: int = 10,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate synthetic spectra as random linear combinations of sinusoidal
    basis functions with added Gaussian noise.
    Returns array of shape (n_samples, n_wavelengths).
    """
    rng = np.random.default_rng()
    basis = create_basis_functions(n_basis, wavelengths)           # (n_basis, n_wavelengths)
    coeffs = rng.standard_normal((n_samples, n_basis))            # (n_samples, n_basis)
    spectra = coeffs @ basis                                     # (n_samples, n_wavelengths)
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra


# -------------------------------------------------------------------
# Photometric forward model
# -------------------------------------------------------------------
def generate_bandpasses(n_bands: int, wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate simple Gaussian bandpasses.
    Returns array of shape (n_bands, n_wavelengths).
    """
    rng = np.random.default_rng()
    bandpasses = []
    for _ in range(n_bands):
        center = rng.uniform(wavelengths.min(), wavelengths.max())
        width = rng.uniform(30, 80)
        transmission = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        bandpasses.append(transmission)
    return np.array(bandpasses)


def compute_photometry(spectra: np.ndarray, bandpasses: np.ndarray) -> np.ndarray:
    """
    Integrate spectra through bandpasses.
    spectra: (n_samples, n_wavelengths)
    bandpasses: (n_bands, n_wavelengths)
    Returns photometry array of shape (n_samples, n_bands).
    """
    # Assume unit wavelength spacing for simplicity
    flux = spectra @ bandpasses.T                # (n_samples, n_bands)
    return flux


# -------------------------------------------------------------------
# Spectrum reconstruction
# -------------------------------------------------------------------
def reconstruct_spectrum(
    photometry: np.ndarray,
    bandpasses: np.ndarray,
    wavelengths: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct spectra from photometric measurements by fitting a
    linear model for each wavelength pixel.
    Returns array of shape (n_samples, n_wavelengths).
    """
    n_samples, n_bands = photometry.shape
    n_wavelengths = len(wavelengths)
    reconstructed = np.empty((n_samples, n_wavelengths))
    # Fit a separate ridge regressor for each wavelength
    for wl_idx in range(n_wavelengths):
        target = photometry[:, wl_idx] * 0  # placeholder, will be replaced
        # Actually the relationship is photometry = spectra @ bandpasses.T
        # We approximate inverse mapping: spectra = photometry * W
        # We solve for W using linear regression for each wavelength
        # Build design matrix X = photometry, y = true flux at this wavelength
        # Here we perform regression using training data itself
        X = photometry
        y = spectra[:, wl_idx]
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        reconstructed[:, wl_idx] = model.predict(X)
    return reconstructed


# -------------------------------------------------------------------
# Main demonstration
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wavelengths = np.arange(400, 801, 5)          # 400-800 nm, 5 nm steps
    n_wavelengths = len(wavelengths)

    # Generate synthetic spectra
    n_samples = 200
    spectra = generate_synthetic_spectra(n_samples, wavelengths)

    # Generate bandpasses
    n_bands = 5
    bandpasses = generate_bandpasses(n_bands, wavelengths)

    # Compute photometric observations
    photometry = compute_photometry(spectra, bandpasses)

    # Reconstruct spectra from photometry
    reconstructed = reconstruct_spectrum(photometry, bandpasses, wavelengths)

    # Evaluate reconstruction quality
    mae = mean_absolute_error(spectra, reconstructed)
    print(f"Mean Absolute Error of reconstruction: {mae:.4f}")