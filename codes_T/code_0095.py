import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """Simple linear combination of Gaussian basis functions."""
    model = np.zeros_like(wavelengths)
    for a, mu, sigma in params:
        model += a * np.exp(-0.5 * ((wavelengths - mu) / sigma)**2)
    return model

def generate_synthetic_spectra(n_samples, wavelengths, rng=None):
    """Generate synthetic spectra from random parameters."""
    rng = rng or np.random.default_rng()
    spectra = []
    true_params = []
    for _ in range(n_samples):
        # Random number of Gaussians (1–3)
        n_gauss = rng.integers(1, 4)
        params = []
        for _ in range(n_gauss):
            a = rng.uniform(0.5, 2.0)      # amplitude
            mu = rng.uniform(wavelengths[0], wavelengths[-1])  # center
            sigma = rng.uniform(5, 30)     # width
            params.append((a, mu, sigma))
        # Sort parameters by center to keep order
        params.sort(key=lambda x: x[1])
        spectra.append(spectral_model(wavelengths, params))
        {True: true_params.append(params), False: None}[True]
    return np.array(spectra), np.array([p for p in true_params])

def create_photometry(spectrum, phot_bin_edges):
    # [x]  &? We repeat …‑?? !...
    (..? ??..?? ??.. …...………. 
...
We need to correct code. Provide complete script. Let's craft minimal reconstruction framework: we will have function to simulate photometry: integrate spectrum over filter transmission defined by bin edges. For simplicity, assume box filters: integral over each bin normalized by width. So photometric vector length equal number of bins. Reconstruction: given photometric values, reconstruct spectrum using linear regression with basis functions of same Gaussian components? But we don't know params. Instead, we can treat reconstruction as approximating spectrum at wavelengths via linear combination of basis set built from all possible Gaussians across training set. Use Ridge regression: training data: synthetic spectra (n_samples, n_wavelengths). Inputs: photometric data (n_samples, n_bins). Fit model: Y_spectra = X_phot @ W . Then reconstruct new spectrum by predicting from its photometry. We'll generate test synthetic spectrum and reconstruct.

Implement functions: generate_synthetic_spectra, create_photometry, fit_reconstruction_model, reconstruct_spectrum.

Let's write code. Ensure import. Use numpy, scipy. We'll not use cross-val.

Return only code. Let's write final answer.assistantfinal code