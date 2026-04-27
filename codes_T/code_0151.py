import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """
    Simple linear combination of Gaussian basis functions.
    
    Parameters
    ----------
    wavelengths : array-like, shape (n_wave,)
        Wavelengths at which to evaluate the model.
    params : array-like, shape (n_basis,)
        Amplitudes for each Gaussian basis function.
        
    Returns
    ----------
    flux : ndarray, shape (n_wave,)
        Computed spectral flux density.
    """
    n_basis = len(params)
    gaussians = []
    for i in range(n_basis):
        mu = 5000 + 400 * i      # centre of each Gaussian
        sigma = 200
        gaussians.append(
            params[i] * np.exp(-0.5 * ((wavelengths - mu) / sigma) ** 2)
        )
    return np.sum(gaussians, axis=0)

def generate_synthetic_spectra(num_spectra, 
                               wavelength_grid,
                               basis_count=100,
                               noise_std=0.02):
    """
    Generate synthetic spectra using random Gaussian‐based
    linear combinations.
        
      *   1   -> .
    
    """
    n_wavel = {??..?????????????????????????????????????????????..???..???...
We need to produce runnable code with functions: define spectral model, generate synthetic spectra, generate photometric data from spectra, reconstruct synthetic spectrum from photometry. Use numpy, scipy, scikit-learn. Minimal implementation.

We'll create a spectral model using basis functions (e.g., Gaussians). Then generate synthetic spectra by random weights. Then photometry via filter curves (simulate maybe using integrals over specific wavelength ranges). Reconstruct using linear regression (Ridge) from photometric features back to full spectrum. We'll need basis representation too: represent spectrum in basis coefficients. We can reconstruct coefficients via regression on photometric values.

Simplify: basis functions: list of basis functions over wavelength grid. Generate spectra as linear combination of basis functions with random weights. Photometric data: compute flux integrated over filters: filter transmission curves defined as simple Gaussian or rectangular windows. For each synthetic spectrum, compute photometric measurements = dot(product basis flux with filter response). Then we train a model: given photometric vector (e.g., 5 bands), predict the coefficients. Use Ridge regression.

Finally reconstruct spectrum: compute coeffs -> spectrum.

Let's write functions:

- create_wavelength_grid(start, end, step)
- create_bases(grid, n_basis): produce matrix of basis functions shape (len(grid), n_basis). Using gaussians.
- generate_synthetic_spectra(n_spec, bases, noise_std): returns flux array shape (n_spec, n_wavelength) and true coeffs.
- create_filters(grid, n_filters): produce transmission arrays shape (n_filters, len(grid)). Simple Gaussians centered at chosen wavelengths.
- compute_photometry(flux, filters): dot product flux * filters along wavelength axis: yields band fluxes.
- train_regressor(photometry, coeffs): use Ridge regression.
- reconstruct_spectrum(regressor, photometry, bases): predict coeffs then compute base*coeff.

Test: generate N=50 training spectra, compute photometry, train, test on new set.

Will produce synthetic example and print shapes etc.

Let's craft code. Ensure reproducibility seed. Use numpy random.

All functions.

Let's write.

Also note we need scikit-learn.

Let's produce final code.assistantfinal code