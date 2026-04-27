import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model: simple Gaussian lines superimposed on a continuum
def spectral_model(wavelengths, amps, centers, widths, continuum_level=0.5):
    """
    wavelengths: array of wavelength values
    amps: list of amplitudes for each Gaussian line
    centers: list of center wavelengths for each Gaussian line
    widths: list of standard deviations for each Gaussian line
    continuum_level: baseline flux level
    Returns an array of fluxes at the given wavelengths.
    """
    flux = continuum_level * np.ones_like(wavelengths)
    for amp, cen, wid in zip(amps, centers, widths):
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid)**2)
    return flux

# 2. Generate synthetic spectra
def generate_synthetic_spectra(num_spectra, wavelengths, amps_range, centers_range,
                               widths_range, noise_std=0.01):
    """
    num_spectra: number of spectra to generate
    wavelengths: array of wavelength values
    amps_range: (min, max) amplitude range for each Gaussian line
    ...
    """
    spectra = []
    true_params = []
    for _ in range(num_spectra):
        amps = np.random.uniform(*amps_range, size=len(amps_range))
        # We want multiple lines; we choose 3 lines for simplicity
        if len(amps_range) == 3:
            centers = np.random.uniform(centers_range[0], centers_range[1], size=3)
            widths = nprandom.choice(???) ??? 
We need to finish code. Let's continue. We'll generate random widths. Use uniform. Ensure to import. Also need generate photometric data from synthetic spectra: sample flux at filter transmission curves. For simplicity use broad-band filters defined by top-hat windows. Then reconstruct spectrum using linear regression on photometric fluxes. Provide functions: generate_filters, get_photometry, reconstruct_spectrum. Use Ridge regression with basis functions (Gaussian lines). But simpler: reconstruct via linear combination of model components? We can treat each Gaussian component as basis. Let's implement:

- basis_fluxes: evaluate spectral model with unit amplitudes for each line? Actually we can treat unknown params as linear coefficients of Gaussian lines plus continuum constant. Use ridge regression to fit from photometric flux vector (size n_filters) to coefficients. Then reconstruct full spectrum.

Thus we need to create design matrix A where each column corresponds to integrated flux of each basis across filters. Basis includes 3 lines + continuum. Compute integrated flux over each filter by integrating model times filter transmission over wavelengths. For synthetic photometry: compute true integrated flux with true params, add noise. Then fit to recover params. Then reconstruct spectrum using recovered params.

Define functions:

- basis_functions(wavelengths): returns array shape (n_points, n_basis)
- filter_transmission(wavelengths, low, high): returns top-hat function.

- generate_filters(num_filters, wavelengths): produce list of transmissions.

- compute_integrated_flux(basis, filters): integrate via trapezoidal rule.

- reconstruct_spectrum(params, wavelengths): use spectral_model.

Ok.

Let's write code fully. Use simple loops. Need to compute integral for each filter and each basis. Use trapz.

Also set random seed for reproducibility.

Let's write now.assistantfinal code