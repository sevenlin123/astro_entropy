import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# --------------------------------------
# 1) Define a simple spectral model
# --------------------------------------
def spectral_model(wavelength, params):
    """
    Simple model: sum of two Gaussian features.
    params: [amplitude1, center1, width1,
             amplitude2, center2, width2]
    """
    amp1, cen1, wid1, amp2, cen2, wid2 = params
    gauss1 = amp1 * np.exp(-(wavelength - cen1)**2 / (2 * wid1**2))
    gauss2 = amp2 * np.exp(-(wavelength - cen2)**2 / (2 * wid2**2))
    return gauss1 + gauss2


# --------------------------------------
# 2) Generate synthetic spectra
# --------------------------------------
def generate_synthetic_spectra(num_spectra, wavelength_grid, true_params):
    """Generate spectra using the spectral model."""
    spectra = []
    for _ in range(num_spectra):
        # add small random noise to parameters
        noisy_params = true_params + np.random.normal(scale=0.1, size=true_params.shape)
        spec = spectral_model(wavelength_grid, noisy_params)
        spectra.append(spec)
    return np.array(spectra)


# --------------------------------------
# 3) Photometric data from synthetic spectra
# --------------------------------------
def make_photometry(spectra, wavelength_grid, filter_wls, filter_trans):
    """
    Compute synthetic photometry by integrating the spectra
    over each filter transmission curve.
    """
    num_filters = len(filter_wls)
    phot = np.zeros((spectra.shape[0], num_filters))
    for i in filter_wls:
        # Find the nearest wavelength indices for the current filter
        idx = np.searchsorted(wavelength_grid, i)
        if idx < len(wavelength_grid):
            # For simplicity, use a top‑horizons 
          .???
We have to finish the code, but it currently got corrupted at the loop. We must correct and complete. The user wants minimal implementation. Use numpy, scipy, sklearn. Provide synthetic example input data inside script. Should return code only.

We need to define filter transmissions, integrate over spectra to get photometric fluxes. Then reconstruct a synthetic spectrum from photometric data using linear regression or inverse problem.

Simplify: use linear combination of basis spectra (e.g., from training set). Use Ridge regression to solve for coefficients that match photometric observations. Reconstruct spectrum as weighted sum.

Let's rewrite entire code cleanly:

- Define wavelength grid, e.g., 400-800 nm, step 1 nm.
- Define true params for model.
- Generate synthetic spectra for n samples.
- Define filters: maybe 5 filters with Gaussian transmission curves centered at 450, 550, 650, 750 nm with widths 50 nm.
- Compute synthetic photometry via integration: flux = sum(spec * trans * delta_lambda).
- Reconstruction: given photometric vector for one spectrum, find coefficients of basis spectra that reproduce photometry. Use ridge regression: X_phot (n_spectra x n_filters), y_phot = target photometry, fit model: X_phot -> coefficients? Wait we want to reconstruct spectrum: we can treat each basis spectrum as feature; need to recover coefficients that best match photometry. In matrix form: P = A * C where A is matrix of photometric responses of basis spectra (n_filters x n_basis). C are coefficients per spectrum. Solve for C: ridge regression minimizing ||P - A C||^2 + lambda ||C||^2. We can compute C for each spectrum: C = (A^T A + lambda I)^-1 A^T P. Or use sklearn Ridge: fit on A^T (but shape?). We'll compute pseudo-inverse manually.

Simplify: choose basis spectra as same as synthetic set minus one sample? For demonstration, take first k spectra as basis.

But easier: Use linear regression directly: we have training set: spectral samples (basis) with known photometric values. Fit a model mapping photometry to spectrum via ridge regression across wavelengths: For each wavelength bin, regress spectrum value vs photometry across samples. This gives weights W (n_filters -> n_wavelengths). So predicted spectrum = W^T photometry + intercept. Use ridge regression separately per wavelength or vectorized using sklearn Ridge with n_targets. That yields reconstruction. Use synthetic dataset.

Implement:

- fit ridge reg: X = photometry (n_samples x n_filters), Y = spectra (n_samples x n_wavelengths). Use Ridge(alpha=1.0, fit_intercept=True). Then predict for new photometry.

- For demonstration: reconstruct first synthetic spectrum using its photometry and compare to original.

Plotting not required. But we can compute error.

Let's write code accordingly.

We'll keep imports: numpy, scipy, sklearn.

We don't need matplotlib unless we want to plot but requirement says no plotting needed. But we might still import for completeness; but we can skip.

Will produce code that runs and prints MSE.

Let's craft final code.

Also ensure functions: spectral_model, generate_synthetic_spectra, make_photometry, reconstruct_spectrum.

Let's code.assistantfinal code