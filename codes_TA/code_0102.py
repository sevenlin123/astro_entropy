import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Lasso

def define_spectral_model(n_points=1000, wavelength_min=300, wavelength_max=2500):
    """
    Returns a simple spectral model grid: wavelengths and basis vectors.
    The basis vectors are Gaussian components spanning the wavelength range.
    """
    wavelengths = np.linspace(wavelength_min, wavelength_max, n_points)
    # Create 5 Gaussian basis functions with different centers
    centers = np.linspace(wavelength_min, wavelength_max, 5)
    sigma = (wavelength_max - wavelength_min) / 20.0
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / sigma)**2)
        basis.append(g)
    return wavelengths, np.array(basis)

def simulate_photometry(mag_phot_band, flux_contribs, wavelengths, band_fluxes):
    **This function?**  
""" 
The plan is...  
????

**... ...…….**
We need to provide final code. Let's design script:

Define spectral model: wavelengths array and basis matrix.

Generate synthetic spectra: choose random coefficients for basis -> true spectrum.

Generate photometric data: define some filter transmission curves (e.g., Gaussian around certain center). Compute synthetic photometry by integrating spectrum times filter over wavelength normalized.

Reconstruct spectrum: using linear regression (Lasso) from photometric data to basis coefficients. Solve X coeff = y where X matrix is integrals of basis over each filter. Reconstruct spectrum as basis @ coeff.

Let's produce code.

Use scikit-learn Lasso.

Implement functions:

define_spectral_model
generate_synthetic_spectra
generate_filters
compute_photometry
reconstruct_spectrum

Add main block generating synthetic data and showing results.

We'll output code. Let's craft.assistantfinal code