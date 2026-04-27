import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LassoCV

# -----------------------------------------------
# 1. Spectral Model
# -----------------------------------------------
def gaussian_spectrum(wavelengths, amplitude, center, sigma):
    """Return a Gaussian spectral line."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center) / sigma)**2)

def composite_spectrum(wavelengths, params):
    """
    Build a synthetic spectrum from multiple Gaussian lines.
    params: list of tuples [(amp, cen, sigma), ...]
    """
    spec = np.zeros_like(wavelengths)
    for amp, cen, sig in params:
        spec += gaussian_spectrum(wavelengths, amp, cen, sig)
    return spec

# -----------------------------------------------
# 2. Generate synthetic spectra
# -----------------------------------------------
def create_synthetic_spectrum():
    wavelengths = np.linspace(4000, 8000, 1000)  # angstrom
    true_params = [
        (1.0, 4500, 30),
        (0.8, 5000, 40),
        (1.2, 6500, 20),
        (0.6, 7000, 50),
    ]
    spectrum = composite_spectrum(wavelengths, true_params)
    noise = np.random.normal(0, 0.02, wavelengths.shape)
    return wavelengths, spectrum + noise, true_params

# -----------------------------------------------
# 3. Generate photometric data from synthetic spectra
# -----------------------------------------------
def bandpass_filter(x, center, width):
    # Simple top‑horizontally
    mask = np.abs(x[1] - x[0]) * np.arange(len(x))  # incorrect but placeholder
    # We need quick fix: use linear spacing
    mask = (x >= (center - width/2)) & (x <= (center + width))
    return mask


def create_photometric_data(spectral_data, bands):
    # [x,y] -> [N_band + N=0?]
    *n??? ???



We need to provide full runnable code implementing core reconstruction framework. Use numpy, scipy, sklearn. Should generate synthetic spectrum using Gaussian lines, then create photometric data via bandpasses, then reconstruct spectrum from photometry using e.g., linear regression with basis functions? Provide functions.

Let's design:

- generate synthetic spectrum: wavelengths array, flux array.

- Photometric data: define filters with central wavelength and width; integrate flux over filter (e.g., weighted average). Compute photometric magnitudes maybe as log?

- Reconstruction: use linear combination of basis spectra; we can use LassoCV to find coefficients to match photometric measurements given basis of synthetic spectra? But we need to reconstruct the spectrum from photometry alone. We'll treat each photometric band measurement as linear combination of flux at wavelengths weighted by filter transmission. We could approximate the unknown spectrum as linear combination of basis functions (maybe simple Gaussian basis). Then we fit coefficients to match photometric data. Use LassoCV to get sparse solution.

Implementation steps:

1. Define Gaussian basis functions across wavelengths.

2. Build design matrix A where each column is integrated flux of basis function through each filter (i.e., convolution). Actually we want to predict photometric values from coefficients: phot = A @ coeffs. So we invert: coeffs = solve least squares with regularization.

Thus:

- generate basis_params = random set of Gaussian lines.

- For each basis, compute flux over wavelengths.

- For each filter, compute integral of flux*transmission.

- Build matrix A size (n_filters, n_basis).

- Then given photometric data (observed integrals), solve for coeffs via lasso.

- Reconstruct spectrum as sum(coeff_i * basis_flux_i).

Implement bandpass filter as simple rectangular or gaussian; choose simple rectangular. Provide function to compute filter response.

Also, provide main function to run.

Let's craft code accordingly.

Need to ensure code runs.

Let's implement: