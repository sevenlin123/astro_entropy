import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def linspace_wavelength(start, end, num):
    """Generate evenly spaced wavelength array."""
    return np.linspace(start, end, num)

def gaussian_filter(wavelength, center, width):
    """Simple Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

# --------------------------------------------------------------------------- #
# Spectral model
# --------------------------------------------------------------------------- #
class SpectralModel:
    def __init__(self, wavelength):
        self.wavelength = wavelength

    def basis_spectra(self, n_bases=3):
        """Create a few synthetic basis spectra."""
        rng = np.random.default_rng()
        bases = []
        for i in range(n_bases):
            # Randomly shaped spectrum with smooth variations
            amp = rng.uniform(0.5, 1.5)
            freq = rng.uniform(0.01, 0.05)
            phase = rng.uniform(0, 2*np.pi)
            base = amp * np.sin(freq * wavelength + phase) + 1.0
            bases.append(base)
        return np.vstack(bases)  # shape (n_bases, n_wave)

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def generate_synthetic_spectra(model, n_spectra=10):
    """Generate synthetic spectra as random combinations of basis spectra."""
    n_bases = model.basis_spectra().shape[0]
    rng = np.random.default_rng()
    weights = rng.uniform(0, 1, size=(n_spectra, n_bases))
    spectra = weights @ model.basis_spectra()
    return spectra, weights  # shapes (n_spectra, n_wave), (n_spectra, n_bases)

def compute_photometry(model, spectra, filters):
    """
    Compute photometric fluxes by integrating each spectrum through each filter.
    `filters` is a list of transmission curves defined on the same wavelength grid.
    """
    n_spectra = spectra.shape[0]
    n_filters = len(filters)
    photometry = np.empty((n_spectra, n_filters))
    for i, filt in enumerate(filters):
        # Integrate spectrum * filter response over wavelength
        photometry[:, i] = np.array([simps(s * filt, model.wavelength) for s in spectra])
    return photometry  # shape (n_spectra, n_filters)

# --------------------------------------------------------------------------- #
# Reconstruction framework
# --------------------------------------------------------------------------- #
class SpectrumReconstructor:
    def __init__(self, model, filters):
        self.model = model
        self.filters = filters
        # Precompute matrix mapping basis weights to photometry
        n_bases = model.basis_spectra().shape[0]
        self.M = np.zeros((len(filters), n_bases))
        for j, filt in enumerate(filters):
            self.M[j, :] = [simps(filt * b, model.wavelength) for b in model.basis_spectra()]

    def reconstruct_weights(self, photometry):
        """Estimate basis weights from photometry via least squares."""
        # Solve M.T @ w = photometry.T for each spectrum
        w_hat = np.linalg.lstsq(self.M.T, photometry.T, rcond=None)[0].T
        return w_hat  # shape (n_spectra, n_bases)

    def reconstruct_spectrum(self, photometry):
        """Reconstruct full spectra from photometry."""
        w_hat = self.reconstruct_weights(photometry)
        return w_hat @ self.model.basis_spectra()  # shape (n_spectra, n_wave)

# --------------------------------------------------------------------------- #
# Main demonstration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Define wavelength grid
    wave = linspace_wavelength(400, 800, 1000)  # nm
    model = SpectralModel(wave)

    # Create filter set (Gaussian passbands)
    centers = [450, 550, 650, 750]
    widths = [30, 30, 30, 30]
    filters = [gaussian_filter(wave, c, w) for c, w in zip(centers, widths)]

    # Generate synthetic spectra
    spectra, true_weights = generate_synthetic_spectra(model, n_spectra=20)

    # Compute photometric observations
    photometry = compute_photometry(model, spectra, filters)

    # Reconstruct spectra from photometry
    reconstructor = SpectrumReconstructor(model, filters)
    reconstructed_spectra = reconstructor.reconstruct_spectrum(photometry)

    # Compare true vs reconstructed spectra
    # Simple mean absolute error per spectrum
    mae = np.mean(np.abs(reconstructed_spectra - spectra), axis=1)
    print("Mean absolute reconstruction error per spectrum:", mae)