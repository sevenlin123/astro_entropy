#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# 1. Define wavelength grid and filter transmission curves
# ----------------------------------------------------------------------
def wavelength_grid(start=400.0, stop=800.0, step=1.0):
    """Return an array of wavelengths in nanometers."""
    return np.arange(start, stop + step, step)


def gaussian_filter(grid, center, width, amplitude=1.0):
    """Create a Gaussian filter transmission curve."""
    return amplitude * np.exp(-0.5 * ((grid - center) / width) ** 2)


def create_filters(grid):
    """Define a small set of broadband filters."""
    filters = {
        "U": gaussian_filter(grid, 365, 35),
        "B": gaussian_filter(grid, 445, 30),
        "V": gaussian_filter(grid, 551, 40),
        "R": gaussian_filter(grid, 658, 50),
        "I": gaussian_filter(grid, 806, 60),
    }
    return filters


# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def synthetic_spectrum(grid, n_components=3, rng=None):
    """Generate a single synthetic spectrum as a sum of Gaussians."""
    if rng is None:
        rng = np.random.default_rng()
    # Random centers within the wavelength range
    centers = rng.uniform(grid[0], grid[-1], size=n_components)
    widths = rng.uniform(20, 80, size=n_components)
    amplitudes = rng.normal(loc=1.0, scale=0.3, size=n_components)
    spec = np.zeros_like(grid)
    for c, w, a in zip(centers, widths, amplitudes):
        spec += a * np.exp(-0.5 * ((grid - c) / w) ** 2)
    # Add low‑frequency continuum
    continuum = rng.uniform(0.1, 0.3)
    spec += continuum
    return spec


def generate_synthetic_spectra(n_samples, grid, rng=None):
    """Generate multiple synthetic spectra."""
    if rng is None:
        rng = np.random.default_rng()
    return np.array([synthetic_spectrum(grid, rng=rng) for _ in range(n_samples)])


# ----------------------------------------------------------------------
# 3. Compute synthetic photometry
# ----------------------------------------------------------------------
def photometry_from_spectra(spectra, filters, grid):
    """
    Compute synthetic photometry.
    Returns a matrix of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    filter_names = list(filters.keys())
    n_filters = len(filter_names)
    phots = np.empty((n_samples, n_filters))
    for i, name in enumerate(filter_names):
        filt = filters[name]
        # Simple band‑integrated flux
        phots[:, i] = simps(spectra * filt, grid, axis=1) / simps(filt, grid)
    return phots


# ----------------------------------------------------------------------
# 4. Reconstruction model
# ----------------------------------------------------------------------
class SpectrumReconstructor:
    """
    Trains a linear mapping from photometric fluxes to full spectra.
    Uses ordinary least squares.
    """

    def __init__(self):
        self.regressor = LinearRegression()

    def fit(self, photometry, spectra):
        """Train the model."""
        self.regressor.fit(photometry, spectra)

    def predict(self, photometry):
        """Predict spectra from photometry."""
        return self.regressor.predict(photometry)

    def score(self, photometry, spectra_true):
        """Return mean squared error between predicted and true spectra."""
        spectra_pred = self.predict(photometry)
        return mean_squared_error(spectra_true, spectra_pred)


# ----------------------------------------------------------------------
# 5. Demo
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # 5.1 Wavelength grid and filters
    grid = wavelength_grid()
    filters = create_filters(grid)

    # 5.2 Generate synthetic data
    n_train = 200
    n_test = 20
    spectra_train = generate_synthetic_spectra(n_train, grid, rng=rng)
    spectra_test = generate_synthetic_spectra(n_test, grid, rng=rng)

    phot_train = photometry_from_spectra(spectra_train, filters, grid)
    phot_test = photometry_from_spectra(spectra_test, filters, grid)

    # 5.3 Train reconstruction model
    reconstructor = SpectrumReconstructor()
    reconstructor.fit(phot_train, spectra_train)

    # 5.4 Predict spectra for test set
    spectra_pred = reconstructor.predict(phot_test)

    # 5.5 Evaluate
    mse = mean_squared_error(spectra_test, spectra_pred)
    print(f"Test MSE of reconstructed spectra: {mse:.4f}")

    # Example: display first test spectrum and its reconstruction
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(8, 4))
    plt.plot(grid, spectra_test[idx], label="True Spectrum")
    plt.plot(grid, spectra_pred[idx], label="Reconstructed", ls="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arb. units)")
    plt.title("Synthetic Spectrum Reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()