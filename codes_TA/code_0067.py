import numpy as np
from scipy import interpolate
from sklearn.linear_model import Lasso
# 1. Spectral Model ---------------------------------------------------------
def build_spectral_model(wavelengths):
    """
    Build a simple linear combination model using Gaussian basis functions.
    """
    # Define 10 Gaussian basis functions evenly spaced across the wavelength range
    centers = np.linspace(wavelengths[0], wavelengths[-1], 10)
    sigma = (wavelengths[-1]-wavelengths[0])/20
    model_basis = []
    for c in centers:
        gauss = np.exp(-((wavelengths-c)**2)/(2*sigma**2))
        normalize = np.linalg.norm(gauss)
        if normalize>0:
            gauss /= normalize
        model_basis.append(gauss)
    return np.array([x[:,None] for x in [i[None,:] for i?]]  # <- incorrect but …
# 2.....
We must output code that meets requirements. Need to fix errors: build_spectral_model returns array of basis vectors. Should produce shape (n_wavelengths, n_bases). We'll create functions:

- build_spectral_model(wavelengths): returns basis matrix.

- generate_synthetic_spectra(n_spectra, basis, coeffs_range): random coefficients within range, compute spectra = basis @ coeffs + noise.

- generate_photometric_data(spectra, filter_transmissions, wavelengths): integrate spectra * filter / integral of filter. Provide fluxes.

- reconstruct_spectrum_from_photometry(fluxes, filter_transmissions, wavelengths, basis): solve linear system for coefficients via lasso or ridge.

Return code.

Also we generate synthetic example.

Let's craft final code. Ensure functions defined.

We will create synthetic filters: e.g., three filters with Gaussian transmissions.

Implementation details:

- wavelengths array e.g., np.linspace(400, 800, 400) nm.

- basis: 10 Gaussians.

- generate_synthetic_spectra: choose n_spectra=5, coeffs uniform random.

- photometric data: compute flux per filter.

- Reconstruction: use Lasso with alpha small.

Return code. Let's produce final answer.assistantfinal