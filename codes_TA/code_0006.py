import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------- 1. Define spectral model -------------------
def spectral_model(wave, coeffs):
    """
    Simple linear combination of basis spectra.
    coeffs : array of length N_basis
    wave   : wavelength grid (1D array)
    Returns flux array over wave.
    """
    # Example basis: polynomials up to order 3
    basis = np.vstack([wave**i for i in range(4)])  # shape (4, len(wave))
    return np.dot(coeffs, basis)

# ------------------- 2. Generate synthetic spectra -------------------
def generate_synthetic_spectra(n_spectra, wave, rng=None):
    """
    Generate n_spectra synthetic spectra using random coefficients.
    """
    rng = rng or np.random.default_rng()
    n_basis = 4
    coeffs = rng.normal(size=(n_spectra, n_basis))
    spectra = np.array([spectral_model(wave, c) for c in coeffs])
    return spectra, coeffs

# ------------------- 3. Generate photometric data from synthetic spectra
def photometric_simulation(spectra, wave, filt_lambdas, filt_sigs):
    """
    1. Gaussian filter transmission curves.
    filters: 
        - wave: array of wavelengths (1D).
        - filt_lambdas: central wavelengths
        - filt_sums: sigma of gaussian.
      -> returns simulated photometric fluxes for each spectrum.
    """
    n_spectrum = spectra.shape[0]
    n_filt = len(filt_lambdas)
    phot = np.zeros((n_spectrum, n_fit))  

    # create Gaussian filters
    filt = [interp1d(wave, np.exp(-0.5*((wave-..-?????)-??????????????????????????? ????)??????????????????????????????????????????????????????????????? ?


We need produce final Python code. We have incomplete code. Let's design a simple reconstruction algorithm.

We need: define spectral model, generate synthetic spectra, generate photometric data, reconstruct synthetic spectrum from photometric data. Use Ridge regression: model flux at wavelength points as linear combination of basis functions derived from photometric constraints. Approach: Suppose we have known filter transmission curves T_i(lambda). Photometric measurement for each filter is integral of S(lambda)*T_i(lambda) d lambda. For synthetic spectra, compute integrals. Then, given photometric data, reconstruct spectrum by fitting coefficients of basis functions such that integrals match photometric data. Use least squares: for each filter, integral equals dot product of coefficient vector with integrals of basis functions times filter. Build matrix A where A[i,j] = integral of basis_j * T_i d lambda. Solve for coefficients via Ridge. Then generate reconstructed spectrum.

Implement functions: basis functions (e.g., polynomials), generate synthetic spectra, generate filters, compute photometric integrals, reconstruct coefficients, reconstruct spectrum.

Need to ensure code runs. Provide synthetic data: wave grid from 400 to 800 nm, 100 points. Filters: maybe 3 filters centered at 450, 550, 650 nm, width 50 nm. Provide Gaussian transmissions.

We'll use numpy integration with trapz.

Let's craft code: