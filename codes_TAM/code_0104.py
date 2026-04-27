import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Spectral model definition ----------------------------------------------------
def spectral_model(wavelengths, params):
    """
    Simple linear combination of Gaussian basis functions.
    wavelengths: array of wavelengths (nm)
    params: list of parameters [amplitude1, center1, width1, amplitude2, ...]
    """
    n_basis = len(params) // 3
    flux = np.zeros_like(wavelengths)
    for i in range(n_basis):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid)**2)
    return flux

# 2. Synthetic spectra generation -----------------------------------------------
def generate_synthetic_spectra(num_specs, wavelengths, true_params_list=None):
    """
    Generate synthetic spectra using the spectral model.
    num_specs: number of spectra to generate.
    wavelengths: array of wavelength points.
    :param true_params_list: optional list of true parameter sets.
    """
    if true_params_list is None:
        true_params_list = []
        for _ in range(num_specs):
            amp = np.random.uniform(0.5, 1.5)
            center = np.random.uniform(500, 700)
            width = np.random.uniform(10, 30)
            params = [amp, center, width]
            true_params_list.append(params)
        # Add more basis if desired
        # -- We keep it single Gaussian per spectrum for simplicity.
    spectra = []
    noise = np.random.normal(scale=0.05, size=(num_specs, 
                                                len(wavelengths))
                                )
    for params in true_params_list:
        spec = spectral_model(wavelengths, params)
        spectra.append(spec)
    spectra = np.array(spectra) + noise
    return spectra, true_params_list

# 3. Photometry extraction from spectra -------------------------------------------------
# Define photometric filter transmission curves
def create_filters():
    filters = {}
    # Filter A: 480-520 nm
    def f1(t):
        return np.where((t >= 480) & (t <= 520), 1.0, 0.0)
    # Filter B; 580–80?; we want...
    let=?????? ???...
We need to finish code. Provide minimal implementation. Let's design simple filters: A and B. Use linear interpolation to compute photometric flux. We'll create filter function arrays. Use wavelengths. We'll compute integrated flux weighted by filter transmission divided by integral of filter. For simplicity.

Then reconstruction: given photometry, reconstruct spectrum via basis functions or ridge regression on wavelengths: we can treat unknown flux at each wavelength as feature vector; photometry is linear combination of flux times filter transmissions. We can set up matrix A where A_ij = integral over filter_j of basis function_i? But simpler: reconstruct via basis of same Gaussians used to generate spectra but with unknown parameters. We'll fit parameters using least squares to match photometric measurements.

Implementation: for each spectrum, we have photometry (two numbers). Solve for params via ridge regression: X matrix: for each filter j compute integral of Gaussian basis function with unknown amplitude etc. But amplitude and center/width unknown too.

Simplify: Instead, we reconstruct spectrum by linear combination of pre-defined basis functions at fixed centers/widths. Eg basis of Gaussians at fixed positions. Then photometry is linear combination of these basis integrals. So we can build matrix M where M[j, i] = integral of basis i through filter j. Then we have y photometric values. Solve for coefficients c via Ridge: c = (M^T M + alpha I)^-1 M^T y. Then reconstruct spectrum as sum c_i * basis_i(wavelengths).

Let's implement. We'll define N_basis=5 Gaussians evenly spaced 450-750 nm width 20 nm. Build filter transmissions: Filter A 480-520, filter B 580-620.

Compute integrals using numerical integration over wavelengths. Use trapz.

Ok. Let's code. Ensure reproducibility. Let's finish.assistantfinal code