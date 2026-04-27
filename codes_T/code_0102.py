import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: a linear combination of Gaussian basis functions
def create_basis_functions(wavelengths, n_components=10):
    """
    Create `n_components` Gaussian basis functions centered uniformly across the wavelength range.
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_components)
    stds = np.full(n_components, (wavelengths.max() - wavelengths.min()) / (4 * n_components))
    # Each basis function is a row in the design matrix
    G = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) ** 2) / stds[None, :]**2)
    return G

def spectral_model(coeffs, basis):
    """
    Evaluate the spectrum given coefficients and basis functions.
    """
    return basis @ coeffs

# 2. generate synthetic spectra
def generate_synthetic_spectra(num_spectra, wavelengths, n_components=10, noise_level=0.02):
    """Create synthetic spectra with random coefficients for each spectral shape."""
    basis = create_basis_functions(wavelengths, n_component=n_components)
    random_coeffs = np.random.randn(num_spectra, n_components)
    spectral = np.array([spectral_model(c, basis) for c in 
                        random_coeffs])
    spectral_noise = spectral + np.random.normal(0, noise_level, spectral.shape)
    return spectral_noise, random_coeffs

# 3. **Photometric data**
    # Let’s calculate the response of a filter band  (i.e., ?).    
    ***

The user wants minimal Python code that implements core reconstruction framework: define spectral model, generate synthetic spectra, generate photometric data from synthetic spectra, reconstruct synthetic spectrum from photometry. Use numpy, scipy, scikit-learn. No extra features. Provide only code.

We need to correct the broken code.

Let's design simple code:

Define wavelengths array e.g., 400-800 nm with 100 points.

Define basis functions: Gaussian mixture.

Generate synthetic spectra: choose random coefficients; compute spectral.

Generate photometric data: filter responses: e.g., define some filters: Gaussian filters with certain central wavelengths and widths. Compute photometric flux by integrating spectrum times filter response over wavelengths.

Reconstruction: Use linear regression (Ridge) to map photometric fluxes to spectrum. We'll treat this as solving for coefficients from photometric data: we have measurement vector y_phot = F * spectrum; we want to estimate spectrum via linear mapping: reconstruct spectrum = basis @ coeff_est. Or directly reconstruct coefficients via linear regression using training set. We can create training set: generate many synthetic spectra and corresponding photometric data; fit ridge regression mapping photometric vector to coefficients; then use it to predict on new photometric data.

We'll produce small example. Steps:

1. Create basis functions G (shape [nwavelengths, ncomponents]).
2. Create random coefficients for training set (ntrain).
3. Generate training spectra: spec_train = G @ coeffs_train.T -> shape [nwavelengths, ntrain].
4. Compute photometric data: define filters: list of filter response arrays (nwavelengths). For each filter, compute integrated flux: filter_response dot spectrum (sum over wavelengths times delta lambda). For each training spectrum, compute photometric vector of length nfilt. So phot_train shape [ntrain, nfilt].
5. Fit ridge regression model: from phot_train to coeffs_train (target). Use sklearn's Ridge.
6. For test sample: generate one synthetic spectrum with random coefficients, compute photometric data.
7. Predict coefficients using ridge model: coeff_pred.
8. Reconstruct spectrum: G @ coeff_pred.
9. Compare to true spectrum: compute error.

Implementation details: wavelengths grid: np.linspace(400, 800, 200) nm.

Filters: let's define 5 Gaussian filters with centers 450, 520, 590, 660, 730 nm, width 50 nm.

Compute filter responses: f = exp(-0.5*(lambda-cent)^2/(width^2)). Optionally normalize.

Compute photometric flux: flux = sum(spec * filter_response) * delta_lambda.

Use delta_lambda = wavelengths[1]-wavelengths[0].

Now code.

Also provide main function to run demonstration.

Return only code. Let's write it.

Be careful to import modules.

Ok.

Let's produce final code.assistantfinal code