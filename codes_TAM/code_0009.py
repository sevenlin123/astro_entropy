import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, a=1.0, b=0.5, c=2000.0):
    """
    Simple Gaussian spectral model: I(λ) = a * exp(-(λ - b)^2 / (2 * c^2))
    """
    return a * np.exp(-((wavelengths - b) ** 2) / (2 * c**2))

def generate_synthetic_spectra(n_samples=10, n_points=1000, seed=42):
    """Generate synthetic spectra using varying Gaussian parameters."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 800, n_points)
    spectra = []
    for _ in range(n_samples):
        a = rng.uniform(0.8, 1.2)
        b = rng.uniform(450, 750)
        c = rng.uniform(30, 80)
        spectra.append(spectral_model(wavelengths, a, b, c))
    spectra = np.array(spectra)  # shape (n_samples, n_points)
    return wavelengths, spectra

def generate_photometry(wavelengths, spectra, filter_centers, filter_widths):
    """
    Convolve spectra with top‑hats for each filter.
    """
    n_filters = len(filter_centers)
    n_samples = spectra.shape[0]
    photometric = np.zeros((n_samples, n_filters))
    for i, (center, width) -> 
?...
We need to output final code. Let's design.

We need functions: spectral_model, generate_synthetic_spectra, generate_photometry, reconstruct_spectrum_from_photometry.

Reconstruction: We can assume we have linear mapping from spectra to photometric via matrix A; then we reconstruct by ridge regression: compute pseudo-inverse or linear regression to invert.

Implementation: Use matrix of basis functions: basis set of synthetic spectra maybe? But simplest: use linear regression with photometric as input, spectra as target. Use Ridge with small alpha.

Steps: 1) generate synthetic spectra (train). 2) compute photometric for same spectra. 3) Train Ridge to predict spectra from photometric. 4) For new photometric data generate synthetic spectrum. But we also need to demonstrate reconstructing a specific one.

Let's create main script that generates dataset, trains model, selects one sample, gets photometry, reconstructs.

Filter: define centers array [500, 600, 700] nm etc width 20 nm.

Convolution: For each filter, integrate intensity over wavelength region where filter passes: top-hat.

Implementation: compute mask = abs(wavelengths - center) <= width/2; photometric = sum(spectrum[mask]*delta_lambda)/sum(mask). Or integral.

Simplify: photometric = sum(spectrum[mask]) / np.sum(mask) maybe.

Now reconstruction: Fit Ridge with X=photometry, y=spectra. Predict for test photometry. Evaluate mean squared error.

Now produce final code. Let's craft.

Also ensure reproducibility.

Let's write: