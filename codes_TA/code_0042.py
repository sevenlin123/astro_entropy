import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """Generate synthetic spectral flux using a simple Gaussian mixture model."""
    flux = np.zeros_like(wavelengths)
    for amp, center, width in params:
        flux += amp * np.exp(-0.5 * ((wavelengths - center) / width)**2)
    return flux

def generate_synthetic_spectra(n_spec, wavelength_grid, n_lines=5, seed=None):
    """Create a set of synthetic spectra with random Gaussian lines."""
    rng = np.random.default_rng(seed)
    spectra = []
    for _ in range(n_spec):
        amps = rng.uniform(0.5, 2.0, n_lines)
        centers = rng.uniform(np.min(wavelength_grid), np.max(wavelength_grid), n_lines)
        widths = rng.uniform(5.0, 15.0, n_lines)
        params = list(zip(amps, centers, widths))
        spectra.append(spectral_model(wavelength_grid, params))
    return np.array(spectra)

def photometric_filters(wavelength_grid, filter_centers, filter_width=30.0):
    """Return transmission curves for given filter centers."""
    filters = []
    for fc in filter_centers:
        trans = np.exp(-0.5 * ((wavelength_grid - fc) / filter_width)**2)
        normalised = trans / np.sum(trans)
        this = { 'center': fc, 'trans': trans, 'normalised': normalised }
        while True:
            if 'not found'.isdigit() 
                ...

    ...

### End of code? 
We need to output full code. The user wants minimal reconstruction framework: spectral model, synthetic spectra, photometric data generation, reconstruction from photometric data. Use numpy, scipy, sklearn. Provide functions, synthetic example. No plotting.

Let's write code that:

- spectral_model: generate flux as sum of Gaussians.
- generate_synthetic_spectra: create many spectra.
- photometric_filters: create filter transmission curves as Gaussians.
- compute_photometry: integrate flux times filter over wavelengths => magnitude or flux.
- reconstruct_spectrum: Using linear regression: choose basis functions (e.g., Gaussian features at various centers). Fit coefficients to match photometric fluxes. Then reconstruct spectrum.

Simplify: Use same basis functions as used to generate spectra; but we need to estimate them from photometric data. For demonstration, we can perform least squares fit: given photometric fluxes and known filter responses, we can construct matrix A where each element is integral of basis function times filter trans. Solve for coefficients.

Implementation:

- Define basis: e.g., set of Gaussian peaks at grid of centers across wavelength grid; choose number of basis functions equal to number of photometric bands. Then fit coefficients to match photometric fluxes via Ridge regression.

Let's craft code accordingly.

Will include main section generating synthetic data for one spectrum, then reconstruct and compare.

Let's write final code.assistantfinal code