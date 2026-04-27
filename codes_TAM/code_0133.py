import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """
    Simple Gaussian absorption line model
    wavelengths: array of wavelength points
    params: list or array of parameters [amp, cen, sigma]
    """
    amp, cen, sigma = params
    return 1.0 - amp * np.exp(-(wavelengths - cen)**2 / (2*sigma**2))

def generate_synthetic_spectra(n_samples, wav_min=400, wav_max=800, n_wav=200):
    """
    Generate synthetic spectra using random Gaussian lines.
    """
    wavelengths = np.linspace(wav_min, wav_max, n_wav)
    spectra = []
    true_params = []
    for _ in range(n_samples):
        amp = np.random.uniform(0.1, 0.5)
        cen = np.random.uniform(500, 700)
        sigma = np.random.uniform(5, 20)
        params = [amp, cen, sigma]
        spec = spectral_model(wavelengths, params)
        spectra.append(spec)
        true_params.append(params)
    return wavelengths, np.array(spectra), np.array(true_params)

def photometry_from_spectrum(spectra, filter_centers, filter_widths):
    """
    Convert each spectr 1D N by Nw (N=..). 
    etc...
    """
This is the conversation; user wants code. They gave incomplete code. We need to finish minimal implementation. Let's construct a script that defines functions: spectral_model, generate_synthetic_spectra, photometry_from_spectrum, reconstruct_spectrum_from_photometry.

We will generate synthetic spectra with Gaussian absorption lines. Photometry: integrate spectra over filter bandpasses: simple top-hat filters.

Reconstruction: Use ridge regression to map photometry to spectrum. We'll create training data from synthetic spectra: features = photometric fluxes; target = full spectrum at wavelength grid. Fit ridge.

Then use reconstruction function: given photometry, predict spectrum via trained model.

Also need main block to generate training set, fit model, test on new sample, compare predicted vs true.

Implement functions accordingly.

Let's write. Ensure reproducibility with seed.

Use numpy arrays.

Use scikit-learn Ridge regression.

For simplicity, we flatten spectrum vector as target.

Implementation steps:

1. spectral_model(wavelengths, params) returns 1 - amp*exp...

2. generate_synthetic_spectra(n_samples) returns wavelengths, spectra matrix (n_samples x n_wav), true_params.

3. photometry_from_spectrum(spectra, filter_centers, filter_widths): For each filter, compute weighted mean flux: integral of spec * filter transmission / integral of filter. We'll approximate via sum over wavelengths: transmission = 1 if |lambda - center| <= width/2 else 0. Compute flux = np.mean(spec[mask]) maybe weighted average.

4. reconstruct_spectrum_from_photometry(photometry, model): predict using ridge model.

Also training: we can split into train/test but we just fit on all.

Define main: generate training data n_train=200. define filters: say 5 filters across range.

Fit ridge.

Test on a new synthetic spectrum: generate one, compute photometry, predict, plot? No plot. Just print errors: RMSE between true and predicted spectra.

Let's write final code.assistantfinal code