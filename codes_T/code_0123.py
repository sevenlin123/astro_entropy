import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model (e.g., linear combination of basis spectra)
def generate_basis_spectra(n_wavelengths, n_components):
    """Generate random basis spectra."""
    return np.random.rand(n_components, n_wavelengths)

def synthesize_spectrum(basis, coeffs):
    """Create a synthetic spectrum as a linear combination of basis spectra."""
    return coeffs @ basis

# 2. Generate synthetic spectra
np.random.seed(42)
n_samples = 5
n_wavelengths = 1000
n_components = 3

basis_spectra = generate_basis_spectra(n_wavelengths, n_components)
true_coeffs = np.random.randn(n_samples, n_components)
synthetic_spectra = np.array([synthesize_spectrum(basis_spectra, c) for c in true_coeffs])

# 3. Generate photometric data from synthetic spectra
# Define filter transmission curves as simple top‑hats
def create_filter(filter_center, width, wavelengths):
    """Create a top‑hat filter transmission curve."""
    filt = np.zeros_like(wavelengths)
    filt[(wavelengths >= filter_center - width / 2) &
         (wavelengths <= filter_center + width / 2)] = 1.0
    return filt

wavelengths = np.linspace(300, 2500, n_wavelengths)
# 4x4? We pick 4 filters
filt_centers = [350, 450, 550, 742]   # B‑like, V‑like 
filt_widths = [100, 100, 100, 100]
filters = [create_filter(c, w, wavelengths) for c in &lt;!<...

We need to output complete runnable code that meets requirements: spectral model, synthetic spectra, photometric data from spectra, reconstruct a synthetic spectrum from photometry. Use numpy, scipy, scikit-learn. Must not include reasoning. Provide only code. Should define clear functions. Must be runnable.

Let's design:

- Generate basis spectra: maybe random or simple sinusoids.
- Synthetic spectra: linear combination with random coefficients.

- Photometric data: given filters (transmission curves), compute band fluxes by integrating spectrum * filter. For simplicity, use discrete integration sum(spectrum * filter) normalized by filter width or sum(filter). We'll just do dot product: flux = sum(spec * filt) / sum(filt).

- Reconstruction: we have observed fluxes (photometric magnitudes) and want to reconstruct spectrum via linear regression: treat basis spectra as features; but here we reconstruct spectrum from photometry: basically we want to infer coefficients from photometry. That is linear model: fluxes = F * coeffs, where F is matrix of filter responses integrated over each basis spectrum: F_ij = sum(basis_j * filt_i) / sum(filt_i). Then we can solve for coeffs via ridge regression (or ordinary least squares). Then reconstruct spectrum by combining basis spectra with estimated coeffs.

Implement functions:

- build_filter_response_matrix(filters, basis_spectra): returns matrix shape (n_filters, n_components).

- estimate_coeffs(ridge, F, photometry): solves ridge regression F @ coeffs ~ photometry. Actually ridge.fit(F, photometry). But we need to solve for coefficients across samples. We have multiple samples: each sample has photometry vector (n_filters). For each sample, we fit ridge separately? Or we can treat all samples at once: F (n_filters x n_components), we want coefficients for each sample. We can compute pseudo-inverse: coeffs = ridge_regression(F.T, photometry.T)? Actually ridge regression for each sample individually: For each sample i, we solve min ||F @ coeff_i - phot_i||^2 + alpha ||coeff_i||^2. This can be solved analytically: coeff_i = (F.T F + alpha I)^(-1) F.T phot_i. So we can precompute M = inv(F.T F + alpha I) F.T. Then coeffs = phot @ M.T? Wait shape: F: (n_filters, n_components). We want M: (n_components, n_filters). Solve for each sample: coeff_i = M @ phot_i. So M = (F.T F + alpha I)^(-1) F.T. So implement function estimate_coeffs(F, photometry, alpha=1.0).

- Reconstruct spectra: reconstructed_spectra = coeffs @ basis_spectra (since coeffs shape (n_samples, n_components), basis_spectra (n_components, n_wavelengths)). We might need to transpose accordingly.

- Then compare original synthetic spectra to reconstructed ones: compute error metrics.

We'll provide synthetic example within script. Should use numpy, scipy, sklearn. But we don't need sklearn if we implement manually. But requirement says use sklearn. We could use Ridge from sklearn to solve per-sample? But ridge.fit expects X, y; but we can treat each sample as y with same X. We could use ridge.fit(F, phot_i) for each sample. But easier manual.

But we can still use Ridge in loops:

for i in range(n_samples):
    ridge.fit(F, photometry[i])
    coeffs[i] = ridge.coef_

However, ridge expects X shape (n_samples, n_features); but here F shape (n_filters, n_components). For each sample, we call ridge.fit(F, phot_i) with X=F (n_filters x n_components), y=phot_i (n_filters). This works.

Implement functions:

- apply_filters(spectra, filters): returns photometry: dot product: for each sample: (spectra * filter).sum() / filter.sum().

- reconstruct_spectra_from_photometry(photometry, filters, basis_spectra, alpha=1.0): compute F matrix, fit ridge for each sample, get coeffs, reconstruct.

- main: generate basis, synthetic spectra, photometry, reconstruct, compute mean absolute error.

Let's craft code.

Let's produce final code snippet. Ensure imports.

Also we must be careful about filter definitions: create_filter.

Ok produce final code.assistantfinal code