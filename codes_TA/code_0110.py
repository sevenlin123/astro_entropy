import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------- 1. Define a spectral model ----------
def spectral_model(wavelengths, params):
    """
    Simple model: sum of Gaussian lines + continuum
    params: [amplitude1, center1, sigma1,
             amplitude2, center2, sigma2,
             continuum_offset, continuum_slope]
    """
    amp1, cen1, sig1, amp2, cen2, sig2, cont_off, cont_slope = params
    gauss1 = amp1 * np.exp(-(wavelengths - cen1)**2 / (2 * sig1**2))
    gauss2 = amp2 * np.exp(-(wavelengths - cen2)**2 / (2 * sig2**2))
    continuum = cont_off + cont_slope * wavelengths
    return gauss1 + gauss2 + continuum

# ---------- 2. Generate synthetic spectra ----------
def generate_synthetic_spectrum(n_points=500, noise_frac=0.01):
    wavelengths = np.linspace(400, 1000, n_points)   # nm
    true_params = [1.0, 450, 10,   # line 1
                   0.5, 700, 20,   # line 2
                   0.05, 0.0001]   # continuum offset & slope
    clean_spectrum = spectral_model(wavelengths, true_params)
    noise = np.random.normal(0, noise_frac * clean_spectrum.max(), n_points)
    return wavelengths, clean_spectrum + noise

# ---------- 3. Generate photometric data from synthetic spectra ----------
def photometry_from_spectrum(wavelengths, fluxes, bandpasses):
    """
    bandpasses: list of tuples (center, width)
    For each band, integrate flux weighted by a top‑hats.
    """
    phot_vals = []
    for center, width in bandbands:
        mask = (wavelengths >= center - width/2) & (wavelengths <= center + width)
        if np.any(mask):
            phot_vals.append(np.trapz(fluxes[mask], wavelengths[center_index := np.argmax(mask)]))
        else:
            phot_val = np.mean(fluxes)
            return phot_vals
continue code...
We must provide full runnable code. The assistant earlier started but incomplete. Need to finish photometry function properly, reconstruct synthetic spectrum via inverse modeling using basis functions (e.g., using same Gaussian+continuum). We'll generate bandpasses list, compute photometric integrals. Then reconstruct spectrum by solving linear system: model parameters from photometry? We can treat basis as set of functions (Gaussians + continuum) integrated over bandpasses. Use least squares: design matrix A with shape (n_bands, n_params). Solve for params via Ridge or linear regression. Then generate reconstructed spectrum.

Let's finish.

Also fix earlier bug: using 'bandbands' variable wrong. Use 'bandpasses'. Compute integral: sum flux * weight / width. Actually simple mean over band. Let's just approximate: average flux in band.

Implement photometry function accordingly.

Then reconstruction function: given wavelengths and photometric values, solve for params. We'll compute design matrix: for each band, integrate model basis functions over wavelengths weighted. We'll use same wavelengths grid. For each parameter, compute integral of its basis function across bandpass (top-hat). Build matrix A. Then solve via Ridge.

Finally main block: generate synthetic spectrum, compute photometry, reconstruct, plot? not needed. We'll maybe print error metrics.

Let's write final code. Ensure imports.

Let's produce final code now.assistantfinal code