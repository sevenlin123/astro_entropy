import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# 1. Define a simple spectral model: linear combination of basis spectra
def create_basis_spectra(n_bases, n_wave):
    """
    Generate n_bases synthetic basis spectra over n_wave wavelengths.
    Each basis is a random Gaussian-shaped profile.
    """
    wavelengths = np.linspace(4000, 8000, n_wave)
    basis = []
    for _ in range(n_bases):
        amp = np.random.rand()
        center = np.random.uniform(4500, 7500)
        width = np.random.uniform(50, 200)
        profile = amp * np.exp(-((wavelengths - center) ** 2) / (2 * width**2))
        basis.append(profile)
    return wavelengths, np.array(basis)

# 2. Generate synthetic spectra
def generate_synthetic_spectrum(basis, coeffs=None):
    """
    Build a synthetic spectrum as linear combination of given basis spectra.
    If coeffs not provided, sample from uniform [0,1].
    """
    if coeffs is None:
        coeffs = np.random.rand(len(basis))
    return np.dot(coeffs, basis), coeffs

# 3. simulate photometric data (a flux across one or more photometric bands)
def photometry_from_spectrum(wavelengths, spectrum, band_center, band_width):
    # Linear interpolation of flux values
    interp_func = interp1d(wavelengths, spectrum, kind='linear', fill_value='extrapolate')
    # 10x? 
    points = np.linspace(bband_center???)????

We need to output complete python code. The assistant truncated. We need to finish.

We need to define functions: photometry_from_spectrum, reconstruct_spectrum_from_photometry. Use Lasso regression for reconstruction using basis spectra projected onto photometric bands. Let's design:

Define band definitions: list of tuples (center, width). Photometric flux per band: integral of spectrum weighted by band response; we approximate as average over band.

Implement photometry_from_spectrum: for each band compute mean flux between center-width/2 to center+width/2. Use interpolation.

Generate synthetic spectra: choose coefficients, produce spectrum.

Generate photometric data: for each synthetic spectrum produce photometric measurements across bands.

Reconstruct: Use basis spectra, compute design matrix A where each column corresponds to flux of each basis in each band. Then solve for coefficients via Lasso with small alpha or zero? Use linear least squares? The core reconstruction: use Lasso or linear regression.

Define reconstruct_spectrum_from_photometry: given photometric vector b, basis wavelengths and basis array, bands, use Lasso to find coeffs, then reconstruct spectrum.

Finally main block: generate basis, generate synthetic spectra (say 5 samples), generate photometric data, reconstruct, compare.

Let's write full code. Ensure proper imports.

Note: In the truncated part, they had "points = np.linspace(bband_center???)???" Should be replaced.

We'll create photometry function properly.

Let's write code: