import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def create_spectral_grid(wavelengths, n_components=5, rng=None):
    """
    Generate random basis spectra (components) over a given wavelength grid.
    Each component is a smooth Gaussian-like function.
    """
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=n_components)
    widths  = rng.uniform(20, 60, size=n_components)
    amplitudes = rng.normal(size=n_components)
    spectra = np.array([amplitudes[i] * np.exp(
        -0.5 * ((wavelengths - centers[i]) / widths[i])**2)
        for i in range(n_components)])
    return spectra  # shape (n_components, n_wavelengths)

def synthesize_spectrum(spectra, coeffs):
    """
    Synthesise an individual spectrum from given component coefficients.
    """
    return np.sum(coeffs[:, None] * spectra, axis=0)

def photometry_from_spectrum(spectrum, bandpasses):
    """
    compute photometric fluxes in each bandpass (integrated flux across 
   (x,y),   so that using these?…  ...... 
We need to produce final code only, no reasoning. Provide python script that implements core reconstruction framework: define spectral model, generate synthetic spectra, generate photometric data from synthetic spectra, reconstruct a synthetic spectrum from photometric data. Use numpy, scipy, scikit-learn. No extra features.

Implement functions:

- generate synthetic spectra: create basis components, random coefficients to produce sample spectra.

- compute photometric fluxes from synthetic spectra: define bandpasses (wavelength ranges). Photometry by integrating flux over bandpass weighted by some filter response; we can approximate with simple top-hat filter.

- reconstruct spectrum from photometric data: we have linear relation between fluxes and coefficients maybe; but we can solve via regression (Ridge). Approach: treat each component as contributing to flux in each band. So build design matrix of how each component contributes to each band. Then fit coefficients using ridge with observed photometric fluxes. Then reconstruct full spectrum using those coefficients and the components.

Implementation steps:

- Define wavelength grid: 400-1000 nm with step 1 nm.

- Generate component basis spectra: each component gaussian.

- For synthetic dataset: choose N spectra (like 10). For each, pick random coefficients from normal distribution.

- Compute photometric data: define bandpasses e.g., Johnson U (350-400nm), B (400-500nm), V (500-600nm), R (600-700nm), I (700-800nm). Actually our grid starts 400 so we adapt.

Better: define bandpasses: [ (400,450),(450,550),(550,650),(650,750),(750,850) ]

For each band, we compute integrated flux: integral of spectrum * filter response (top-hat). We'll just sum spectrum over wavelengths in that band.

- Reconstruct: Build response matrix A where each column corresponds to component's integrated flux in each band: integrate each component over each band. Then we observe vector of photometric fluxes y = A * coeffs + noise. Fit coeffs via ridge regression: ridge.fit(A, y). Then reconstruct spectrum: sum(coeffs * component).

Also we may include standard scaler to normalize.

Will output prints.

Also provide test run.

Let's craft final code.

Ensure we use numpy, scipy, sklearn. Provide functions:

- create_basis_spectra

- synthesize_spectrum

- compute_photometry

- reconstruct_spectrum

- main demonstration.

Edge: Use rng for reproducibility.

Implement bandpasses as list of tuples.

Make code self-contained and runnable.

Also we must adhere to requirement: "Do NOT show reasoning or thinking. Do NOT explain the code." So output code only.

Let's produce final code.assistantfinal code