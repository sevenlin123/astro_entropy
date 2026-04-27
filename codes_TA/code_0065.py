import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wave):
    """
    Simple model: sum of two Gaussian features plus a linear continuum.
    """
    g1 = np.exp(-0.5 * ((wave - 5000) / 200)**2)
    g2 = 0.8 * np.exp(-0.5 * ((wave - 8000) / 300)**2)
    continuum = 1 + 0.0001 * (wave - 4000)
    return g1 + g2 + continuum

def generate_synthetic_spectra(n_spectra=100, wave_min=4000, wave_max=9000,
                               n_points=1000):
    """
    generate random spectral shape parameters.
      - amplitude scaling factors for each component.
    the first part: (a1,a2/scale, etc..). 
- [..]. 
        &??

We have to write full code. Let's craft functions: generate synthetic spectra with random scaling for Gaussian components and continuum slope. Then compute photometric fluxes using filter transmission curves (synthetic filters). Reconstruct spectrum via linear regression (Ridge) using filter responses as basis functions? We can approximate spectrum as combination of basis functions that are the model functions evaluated at wavelengths times filter transmissions? We can perform deconvolution: given known filter response curves (e.g., Gaussian centered at certain wavelength), we solve for coefficients in basis of those filter responses. We'll generate synthetic filters: maybe 5 filters. For each filter, we integrate synthetic spectrum weighted by filter response to get flux. Reconstruction: assume spectrum is linear combination of basis functions equal to each filter's weighted kernel across wavelengths? Actually we can reconstruct by performing linear interpolation of photometric fluxes using precomputed mapping between filters and spectrum basis. Might use Ridge regression: treat each pixel as a target variable, inputs are filter fluxes. Use training set of synthetic spectra and computed photometry, fit a linear model for each wavelength. Then apply to new photometry to predict spectrum.

Let's implement:

- function compute_fluxes(spectrum, wave, filters): returns flux per filter by integral of spectrum * filter_response / integral of filter_response (to normalize).

- generate synthetic filters: each filter defined by central wavelength and width; response as gaussian.

- generate training dataset: many synthetic spectra with random scaling for two Gaussians and continuum slope; compute photometry; store spectra.

- Fit linear models for each wavelength: For each wavelength bin, build ridge regression with input photometry (vector length n_filters) to predict flux at that wavelength.

- Then reconstruct spectrum from new photometry: feed into models to predict flux array.

Implement main part: generate training, fit models, generate a test spectrum, compute its photometry, then reconstruct and plot? They said no plotting. Just compute reconstructed spectrum.

Return code.

Let's write.

Also we need imports: numpy, scipy, sklearn.linear_model.Ridge.

Ensure reproducibility: set random seed.

Implementation details:

- wave grid: np.linspace(wave_min, wave_max, n_points)

- Filters: create dictionary name->response array same size as wave.

Define filter params: centers=[4500,5500,6500,7500,8500]; widths=300 each; responses: exp(-(wave-center)^2/(2*width^2)).

Normalization: responses already.

Compute flux: integral over wave of spec * response / integral of response.

Spectra generation:

- random amplitudes for g1, g2: between 0.5 and 1.5

- continuum slope random: between -0.0002 and 0.0002; baseline 1.

Add noise? Not necessary.

Return arrays.

Fit models:

- X: shape (n_spectra, n_filters)

- Y: shape (n_spectra, n_points)

For each wavelength index j: fit Ridge(alpha=1.0) on X -> y[:,j]

Store models list of length n_points.

Reconstruction: Given photometry vector, predict flux for each wavelength: [model.predict([flux])[0] for model in models].

Ok.

Return script.

Let's craft code.assistantfinal code