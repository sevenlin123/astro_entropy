import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Define a simple spectral model: Gaussian basis functions
def gaussian_basis(wave, centers, sigma):
    return np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / sigma) ** 2)

# Generate synthetic spectra using random coefficients
def generate_synthetic_spectra(n_spectra, wave, centers, sigma, noise_std=0.05):
    n_centers = len(centers)
    coeffs = np.random.randn(n_spectra, n_centers)
    spectra = gaussian_basis(wave, centers, sigma) @ coeffs.T
    noise = np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

# Integrate spectra over a filter transmission curve to get photometry
def integrate_over_filter(spectra, wave, filter_wave, filter_trans):
    # interpolate filter response on spectrum wavelength grid
    filt_interp = np.interp(wave, filter_wave, filter_trans, left=0, right=0)
    # integrate using Simpson's rule (assuming uniform spacing)
    phot = simps(spectra * filt_interp, wave, axis=1)
    return phot

# A simple photometric reconstruction using Ridge regression
def ridge_reconstruction(phot, filter_wave, filter_list, wave, sigma=2.0):
    # Build design matrix by integrating basis functions over each filter
    X = []
    for filt in filter_list:
        filt_coeffs = []
        for c in filt['center']:
            # The basis function integral over filter
            filt_integral = simps(np.exp(-0.5*((wave - c)/sigma)**2) * filt['trans'], wave)
            # add to list
            filt_coeffs.append(filt_integral)
        X.append(filt_coeffs)
    X = np.array(X).T  # shape (n_samples, n_filters)
    # Fit Ridge regression
    clf = Ridge(alpha=1.0)
    clf.fit(X, phot)
    # Predict coefficients for new data
    coeffs = clf.predict(X)
    # Reconstruct spectra using basis functions
    basis = gaussian_basis(wave, centers=centers, sigma=sigma)
    reconstructed = basis @ coeffs.T
    return reconstructed, coeffs

def main():
    # Define wavelength grid
    wave = np.linspace(5000, 10000, 300)  # wavelengths in Ångstroms
    # set centers for Gaussian basis
    centers = np.linspace(5500, 9500, 10)
    # synthetic spectra
    n_spectra = 50
    spectra, true_coeffs = generate_synthetic_spectra(
        n_spectra, wave, centers, sigma=200, noise_std=0.02
    )

    # Define filters
    filter_dicts = [
        {
            u"filter_name":   "u",
            f"{...}???"

We need to finish the code. The user wants minimal code implementing core reconstruction framework. We should define spectral model, generate synthetic spectra, generate photometric data, reconstruct a synthetic spectrum from photometry. Use numpy, scipy, scikit-learn. Not need LOOCV, etc. Should be runnable Python code with clear functions. Must produce synthetic example input data inside script. No plotting needed but can import matplotlib (but not used). But we can skip plotting to keep minimal.

We need to define a filter set. Could use simple top-hat filters: e.g., define filter transmission curves: center wavelengths and width. For each filter, we generate filter_wave and filter_trans arrays.

Simpler: choose few filters, e.g., 5 filters. For each filter, define transmission function over wave grid: e.g., Gaussian filter centered at some central wavelengths. Then integrate.

We then reconstruct spectrum from photometry using a simple method: maybe linear regression mapping photometric fluxes to coefficient vector of basis functions. Actually we could use the forward modeling approach: we can compute the mapping matrix between basis coefficients and photometric fluxes: each photometric measurement is sum over basis integrals times coefficient. So we can solve for coefficients via linear regression (Ridge). That yields reconstructed spectrum.

We need functions:

- gaussian_basis(wave, centers, sigma): returns array shape (len(wave), len(centers))

- generate_synthetic_spectra(n_spectra, wave, centers, sigma, noise_std): returns spectra (n_spectra, len(wave)) and true coefficients (n_spectra, len(centers))

- define_filters(): returns list of dicts each containing 'name', 'center', 'width', maybe 'trans' computed on wave grid.

- compute_photometry(spectra, wave, filters): returns phot array shape (n_spectra, n_filters)

- reconstruct_spectrum_from_photometry(phot, wave, centers, sigma, filters): uses ridge regression to map phot to coefficients; then reconstruct spectrum.

Also optionally we can compute error metrics. But not required.

We need to ensure the code runs: define main() function that calls everything and prints results. Provide synthetic data generation.

Let's write.

We will import numpy, scipy.integrate.simps, sklearn.linear_model.Ridge. We'll also import random seed for reproducibility.

Implementation details:

- For filters: we can create simple top-hat filter: trans=1 within width +- half-width else 0. Or use Gaussian filter: trans = exp(-0.5*((wave-filter_center)/filter_width)**2). Let's use Gaussian filter for simplicity.

- For integration: we integrate spectra*filter_trans over wave to get phot flux. Use simps along axis=1.

- For reconstruction: we know mapping matrix M where each element M[i,j] is integral over wave of basis_j * filter_i trans. So we can compute design matrix X = M. Then given phot vector y, we estimate coefficients c = ridge.fit(X, y). But since X does not depend on sample, we just use ridge.fit(X, y) for each sample. Actually we have multiple samples, so X is same for all samples. We fit ridge on training data: we can do Xc = ridge.fit(X, y). But ridge expects X shape (n_samples, n_features). Here we have Y shape (n_samples,) per filter? Wait: Phot is shape (n_samples, n_filters). For each filter separately, we can solve c vector length n_centers via ridge regression using X of shape (n_filters, n_centers). Actually we want to predict c (coeff vector) from phot fluxes. So our target is c (n_centers). Our predictor is phot (n_filters). So we can fit Ridge regression with design matrix phot (n_samples, n_filters) and targets true_coeffs (n_samples, n_centers). But this would involve solving for each center coefficient separately via linear regression. That might be easier: for each center j, we fit ridge: coeff_j ~ phot. Equivalent to doing ridge fit with X=phot, y=true_coeffs[:, j]. So we can fit one ridge model per coefficient or use multi-target Ridge via sklearn.multioutput.MultiOutputRegressor. But simplest: use Ridge with multi-target capability: Ridge(n_targets=n_centers). Actually sklearn's Ridge supports multi-output regression by default. So we can fit ridge.fit(phot, true_coeffs). Then we can predict coefficients: pred_coeffs = ridge.predict(phot). Then reconstruct spectra: basis @ pred_coeffs.T (since basis shape (n_wave, n_centers)). Then we can compare with original.

But to reconstruct, we may not have true_coeffs in practice. In reconstruction step, we just need to get predicted coefficients from phot using ridge trained on some training data (maybe same synthetic data). But we can fit on training data (like same dataset) and then predict on test data. For simplicity, we can split into train/test.

Let's implement:

- generate_synthetic_spectra(...)

- define_filters(wave, n_filters=5): returns list of filter dicts each with 'name', 'center', 'width', 'trans' computed.

- compute_photometry(spectra, wave, filters): returns phot.

- reconstruct_spectrum_from_photometry(phot, wave, centers, sigma, filters): uses design matrix M to build forward mapping X for ridge. But better: we can directly use ridge regression on phot->coeffs. So we don't need M. But we can compute M to verify.

We'll implement: