import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# ------------------------------------------------------------------
# 1. Define a simple linear spectral model
# ------------------------------------------------------------------
def spectral_model(wavelengths, params):
    """
    A toy linear model for a spectrum:
        spec(λ) = Σ_i p_i * basis_i(λ)
    where basis_i are Gaussian components.
    """
    n_comp = params.shape[0]
    spectrum = np.zeros_like(wavelengths)
    for i in range(n_comp):
        amp, cen, wid = params[i]
        spectrum += amp * np.exp(-0.5 * ((wavelengths - cen) / wid)**2)
    return spectrum

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(num_spectra=50, seed=42):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(3500., 7500., 1000)  # Angstroms
    # random parameters: amplitude [0,1], center [3800,7200], width [100,400]
    param_list = []
    for _ in range(num_spectra):
        amps = rng.uniform(0.1, 1.0, size=3)
        cents = rng.uniform(3800, 7200, size=3)
        wids = rng.uniform(100, 400, size=3)
        params = np.vstack([amps, cents, wids]).T
        param_list.append(params)
    spectra = np.array([spectral_model(wavelengths, p) for p in param_list])
    return wavelengths, spectra, param_list

# ------------------------------------------------------------------
# 3. Create photometric filters
# ------------------------------------------------------------------
def create_filters():
    """
   Creates a set of simple trapezoidal filter curves.
      w: [...]
    .... .... ...
    """
    def filter_shape(shift=0., width=1200., intensity=0.1):
        mid = 5200 + shift
        """?????????????..?????"""
continue writing...
We need to output final code. Let's design simple filter curves: trapezoidal 4 bands: u,g,r,i etc. We'll just define some wavelength ranges. Then compute photometry by integrating spectrum times filter response normalized.

Then reconstruction: given photometric fluxes, reconstruct spectrum by solving linear inverse with ridge regression using basis functions. Use same spectral model basis as earlier; treat each component as variable to fit to photometric data across filters. Solve for coefficients that best reproduce photometry.

Let's produce functions: create_filters -> returns list of (center,wave array,response). Simpler: define bandpasses as arrays.

Implement photometry: given wavelengths and spectrum, filter response: integral(spectrum*response) / integral(response). We'll produce vector.

Reconstruction: given photometric vector (length m filters), we build matrix A where A[j,i] = flux predicted from basis component i in filter j. Solve ridge regression.

Let's implement.

Also synthetic example usage in main block.

Return code.assistantfinal code