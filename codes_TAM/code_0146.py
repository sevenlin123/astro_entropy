import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define a spectral model
# ----------------------------------------------------------------------
def spectral_model(wavelengths, params):
    """
    Simple Gaussian spectral model with amplitude, center, width.
    params: [amplitude, center, sigma]
    """
    amp, cen, sigma = params
    return amp * np.exp(-(wavelengths - cen) ** 2 / (2 * sigma ** 2))

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectrum(n_spec, wav_grid, rng=None):
    """Generate n_spec random Gaussian spectra on given wavelength grid."""
    if rng is None:
        rng = np.random.default_rng()
    spectra = []
    true_params = []
    for _ in range(n_spec):
        amp = rng.uniform(0.5, 1.5)
        cen = rng.uniform(4000, 7000)
        sigma = rng.uniform(50, 150)
        true_params.append([amp, cen, sigma])
        spectra.append(spectral_model(wav_grid, [amp, cen, sigma]))
    return np.array(spectra), np.array(true_params)

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------      
def photometric_response(wav_grid, band_edges):
    """
    Calculate transmission profile for each band as a top-hat function.
    
    :param wav_grid: wavelengths in angstroms.
    :int???????..???????????????????????????...?? ??? ??????
    return :   ->?????????????????..??? ?????? ?????....
        *????????????????..????..??..????
    """
    
    return 2???.......

This seems incorrect. Let's produce correct script...
We need to provide full working code. Let's craft functions:

Define spectral model: maybe multiple Gaussians.

Generate synthetic spectra: random parameters, create spectra array.

Photometric data: convolve spectra with filter response functions (top-hat). Provide band edges list.

Reconstruct spectrum: use linear regression using basis functions? Use ridge regression on fluxes to predict spectrum across grid. We can express spectrum as combination of basis functions (e.g., gaussian kernels centered at wavelengths). Use ridge regression training set: we have synthetic spectra and photometric fluxes. Solve for weights to approximate spectral shape.

Implement reconstruct_spectrum(fluxes, basis, filter_responses, wav_grid): solve ridge regression.

Let's implement:

basis: e.g., Gaussian kernels at selected wavelengths.

filter_response: for each band, compute matrix M where M_ij = integral of basis_j * response_i over wavelength divided by integral of response_i (i.e., photometric flux). For simplicity, top-hat filters with uniform transmission.

Then given flux vector f (size nbands), we have f ≈ M w, where w are basis weights. Solve ridge regression: w = (M^T M + alpha I)^-1 M^T f.

Then reconstructed spectrum = sum_j w_j * basis_j(wav_grid).

Provide demonstration.

Let's code.

Also need to generate synthetic training set: many spectra, compute their photometry using same filter responses.

Use ridge regression training: we have matrix A (nbands x nspec) for photometry, B (n_basis x nspec) for basis coefficients (obtained by fitting each spectrum to basis via least squares). Then we learn mapping from photometry to basis coefficients via ridge regression: W = (A A^T + alpha I)^-1 A B^T? Wait typical mapping: For each basis coefficient vector c_s (length n_basis) and photometric vector f_s (length nbands), we want linear map L such that c_s ≈ L f_s. So solve L minimizing ||C - L F||^2 + reg. L shape: n_basis x nbands. Solve via ridge: L = C F^T (F F^T + alpha I)^-1.

Then given new photometry, compute coeffs = L f_new, reconstruct.

Let's implement.

Simplify: choose n_basis=20 gaussian kernels across wavelength grid.

Let's code. We'll include simple main block generating data, training, testing, printing RMS error.

Return code only.assistantfinal code