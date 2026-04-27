import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------- 1. Define a spectral model -------------------
def spectral_model(wavelengths, params):
    """
    Simple linear combination of Gaussian basis functions.
    wavelengths: array of wavelengths (nm)
    params: coefficients for each Gaussian component
    Returns the spectrum (intensity) evaluated at given wavelengths.
    """
    # Define fixed Gaussian centers and widths
    centers = np.array([400, 500, 600, 700])
    widths  = np.array([20, 20, 20, 20])
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for coeff, center, width in zip(params, *np.meshgrid(centers, widths)):
        spectrum += coeff * np.exp(-0.5 * ((wavelengths - center)/width)**2)
    return spectrum

# ------------------- 2. Generate synthetic spectra -------------------
def generate_synthetic_spectra(n_samples, rng=None):
    """
    Generates synthetic spectra by sampling random coefficients.
    n_samples: number of spectra to generate
    returns: (samples, spectra)
    """
    rng = np.random.default_rng(rng)
    # Random coefficients in range [0, 1]
    coeffs = rng.uniform(low=0.0, high=1.0, size=(n_samples, 4))
    wavelengths = np.linspace(350, 800, 1000)
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return coeffs, spectra, wavelengths

# ------------------- 3. **Photo**–reconstruction ...
def generate_photometric_data(spectra, wavelengths, filters):
    """
   The function create a photometric filter band‐intensity 
   *spectrum … … .. .. ..?…???-???  …  
Let's finish. Need photometric filters defined as transmissions. Use simple top-hat. Then get photometric flux by integrating product. Reconstruction: use ridge regression to map photometry to spectrum. Implement functions: compute_photometry, reconstruct_spectrum. Provide example main. Let's craft code.assistantfinal