import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, coeffs):
    """
    A simple linear combination of Gaussian basis functions.
    wavelengths: array of wavelengths
    coeffs: coefficients for each basis function
    Returns the modeled spectrum.
    """
    # Define three Gaussian basis functions with fixed centers and widths
    centers = np.array([400, 550, 700])  # nm
    widths = np.array([50, 70, 90])      # nm
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for c, w, a in zip(centers, widths, coeffs):
        spectrum += a * np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return spectrum

def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Generate synthetic spectra using random coefficients.
    n_samples: number of spectra.
      wavelengths: array of wavelength points.
    """
    coeffs = np.random.rand(n_samples, 3)  # random coefficients for 3 basis functions
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return spectra, coeffs

def photometric_filter(
    phot_wavelengths,
    phot_indices,
    spectral_response,
    fwhm=120,
    th=1e-8,
    ρ=10,
    gini: int = 0,
   ... ? 
We have incomplete code. Need to produce minimal working script. Let's design simple filter: convert spectra to photometric flux by integrating over filter bandpasses (simple top-hat). We'll create filter curves: center wavelengths and width. Then compute photometric measurements by dot product.

Reconstruction: use ridge regression to map photometric vector to spectral coefficients then reconstruct.

We'll produce synthetic data: wavelengths 400-800 nm 100 points. Photometric bands at 450, 550, 650 nm. Use top-hat filter width 50 nm.

Implement photometric data generation: For each spectrum, integrate with filter response (array of same length). Use Simpson integration or trapezoidal.

Reconstruct: given photometric values and known filter responses, solve for coeffs via ridge regression. Use training data to fit mapping between photometric and coeffs? Actually we can directly invert using linear algebra since we have linear relation: phot = R @ coeffs where R is matrix of integrals of basis functions through filters. So we can precompute R and use least squares.

Simpler: build matrix M where each column is integral of each basis function over each filter. Then solve for coeffs via least squares for each sample.

Then reconstruct spectrum by spectral_model with predicted coeffs.

Let's write functions: filter_response, generate_photometry, reconstruct_coeffs, reconstruct_spectrum.

Return main block generating data, reconstructing, compare.

Now produce final code.assistantfinal code