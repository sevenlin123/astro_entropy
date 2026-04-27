import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# Define wavelength grid (nm)
lam = np.arange(350, 1100, 5)

# Spectral model: Gaussian peaks for absorption features
def spectral_model(params, lam):
    """params: [amp1, cen1, sig1, amp2, cen2, sig2, offset]"""
    amp1, cen1, sig1, amp2, cen2, sig2, offset = params
    return (
        amp1 * np.exp(-((lam - cen1)**2) / (2 * sig1**2))
        + amp2 * np.exp(-((lam - cen2)**2) / (2 * sig2**2))
        + offset
    )

# Generate synthetic spectra
np.random.seed(42)
n_spectra = 50
true_params = np.column_stack(
    [
        np.random.uniform(0.8, 1.2, n_spectra),      # amp1
        np.random.uniform(450, 550, n_spectra),     # mean1
        np.random.uniform(10, 30, n_sPoetry???)
We need to produce correct code: generating synthetic spectra, photometric data via filter transmissions, then reconstruct using linear regression with basis functions from known model? They said ignore validation, just core framework. We need synthetic example input data inside script. The reconstruction step could be simple: given photometric fluxes, we can fit coefficients of base spectral components by minimizing difference between predicted photometric integrals and measured ones. Use linear regression between filter responses and spectral basis functions. We'll create filter response curves: e.g., three filters: U,B,V.

Implementation steps:
1. define lam array.
2. define spectral_model function that returns synthetic spectrum given parameters.
3. generate synthetic spectra: for each object, random parameters within ranges, compute spectrum.
4. generate photometric data: define filter curves (Gaussian or box). For each spectrum, integrate with filter response to get fluxes.
5. Reconstruction: treat each filter measurement as linear combination of spectral basis functions? Actually we can approximate unknown spectrum by linear combination of pre-defined basis functions (e.g., set of Gaussian templates). Fit coefficients by solving linear least squares using filter integrals as constraints. Equivalent to building matrix A where A_{ij} = integral of basis_i * filter_j; given photometric vector p, solve for coeffs c via linear regression.

Let's write code accordingly.

Let's produce base spectra: we can use same spectral_model with fixed parameters? For reconstruction we will use same set of basis functions used in generation? Maybe use same param sets as training.

Simpler: choose some basis set: e.g., 3 Gaussians with fixed centers and sigmas; use LinearRegression to solve for coefficients from photometric fluxes.

Implementation:

Define basis functions: gauss1, gauss2, gauss3 with centers 500,650,800 nm widths 20 nm.

For each synthetic spectrum, we can generate true coefficients using random values.

Compute full spectrum: sum(coeff_i * basis_i)+offset.

Generate photometric data: filter responses: U (center 360, width 40), B (440, 80), V (550, 90). For each filter, compute integral of spectrum*filter_response over lam, divided by integral of filter_response.

Reconstruction: given photometric values p, we set up design matrix M where M_{j,i} = integral(basis_i*filter_j). Solve for coeffs via linear regression: c = (M^T M)^{-1} M^T p.

Then reconstruct spectrum: sum(c_i*basis_i).

Return code. Use numpy, scipy, sklearn linearregression. Not necessary to use cross-validation.

Let's craft code. Need to import simps for integration. We'll implement gaussian filter function. Provide everything in one script.

Let's produce final code.assistantfinal code