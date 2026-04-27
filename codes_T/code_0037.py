import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# 1. Define spectral model
# ---------------------------------------------
def gaussian_profile(wavelength, amp, center, sigma):
    """Single Gaussian component."""
    return amp * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)

def composite_spectrum(wavelengths, params):
    """
    Build a composite spectrum from multiple Gaussian components.
    params: list of tuples (amp, center, sigma) for each line
    """
    spec = np.zeros_like(wavelengths)
    for amp, center, sigma in params:
        spec += gaussian_profile(wavelengths, amp, center, sigma)
    return spec

# ---------------------------------------------
# 2. Generate synthetic spectra
# ---------------------------------------------
def generate_synthetic_spectra(n_spec, wavelengths):
    """
    Create n_spec synthetic spectra with random Gaussian lines.
    Each spectrum has 2-5 lines.
    """
    spectra = []
    true_params = []
    rng = np.random.default_rng()
    for _ in range(n_spec):
        n_lines = rng.integers(2, 6)
        amps = rng.uniform(0.5, 1.5, size=n_lines)
        centers = rng.uniform(wavelengths[0], wavelengths[-1], size=n_lines)
        sigma = rng.uniform(5.0, 15.0, size=n_lines)
        params = list(zip(amps, centers, sigma))
        spectra.append(composite_spectrum(wavelengths, params))
        true_params.append(params)
    return np.array(spectra), true_params

# ---------------------------------------------
# 3. Generate photometric data
# ---------------------------------------------
def integrate_over_band(wavelengths, spectrum, band_edges):
    """
    Integrate a spectrum over a filter band using Simpson's rule.
    band_edges: tuple (wl_min, wl_max)
    """
    mask = (wavelengths >= band_edges[0]) & (wavelengths <= band_edges[1])
    if not mask.any():
        return 0.0
    return simps(spectrum[mask], wavelengths[mask])

def create_photometric_points(spectra, wavelengths, band_list):
    """
    For each spectrum, calculate photometric fluxes across bands.
    band_list: list of tuples (band_name, (wl_min, wl_max))
    """
    photom = []
    for spec in spectra:
        fluxes = [integrate_over_band(wavelengths, spec, band[1]) for band in 
                  (band_list)]
        photom.append(fluxes)
    return np.array(photom)

# ---------------------------------------------
# 4. Spectrum reconstruction
# ---------------------------------------------
def reconstruct_spectrum_from_photometry(photometry, band_edges, wavelengths):
    """
    Given photometric fluxes and band edges, build linear system to 
    solve for spectrum via least‑squares.
    A matrix A: each row corresponds to a band integration
    **A[i,j] = integral of basis j over band i**
    */
    def basis_function(j, w):
        """We use a simple piecewise constant basis per wavelength point."""
        basis = np.zeros_like(w)
        start, end = int(round((i * 0.1)) * 1?;?? ?????? 
...
We are ChatGPT. The user wants minimal python implementation of core reconstruction framework. We must provide code that:

- defines a spectral model
- generates synthetic spectra
- generates photometric data from synthetic spectra
- reconstruct a synthetic spectrum from photometric data

Using numpy, scipy, scikit-learn. Provide clear functions. No validation methods. No plotting. Provide synthetic example input data inside script. Don't show reasoning or explanation. Only code. So we need to produce final python code. The previous assistant gave partial code but incomplete. We need to provide complete code. Let's design algorithm:

Define spectral model: maybe use Gaussian components. Provide function composite_spectrum.

Generate synthetic spectra: create random parameters.

Generate photometric data: integrate over filter bands.

Reconstruct synthetic spectrum from photometry: Use linear regression to approximate spectrum given photometric fluxes and precomputed basis functions for each wavelength. We can treat each wavelength as a coefficient to fit.

Simplify: Use least squares linear regression to solve for spectrum vector given integrals. Suppose we have M photometric bands, N wavelengths. Build matrix A (M x N). Each element A[i, j] = integral over band i of basis function at wavelength j. But basis functions may be delta at each wavelength or something like piecewise constant. Simplest: treat each wavelength bin as independent variable. Then for each band, integral over that band is sum of flux at each wavelength times width of that bin within band. That yields linear relation: flux_i = sum_j weight_ij * spectrum_j, where weight_ij is integration weight (wavelength bin width) if wavelength bin lies within band else 0. So we can compute A by weights. Then we can solve least squares: spectrum = pinv(A) @ photometry. Or use LinearRegression from scikit-learn: Fit X=A, y=photometry for each spectrum. But we want to reconstruct spectrum for each sample individually. Actually each spectrum produces photometric vector y. We can reconstruct each by solving linear system individually: use numpy.linalg.lstsq.

So functions:

- build_weight_matrix(band_edges, wavelengths): returns A (M x N).

- reconstruct_spectrum(photometry_vector, A): uses lstsq.

Generate synthetic data: choose n_samples. Provide band list e.g. 5 bands with defined edges. Generate synthetic spectra of length N. Compute photometric fluxes. Then reconstruct each. Evaluate error. Print mean absolute error.

Now produce code accordingly.

We'll keep everything in one script. Provide main guard.

Ok.

Let's write code.

Will produce minimal. Use numpy arrays.

Potential issues: Simpson integration etc. But for reconstruction we can use direct weighted sum.

We'll compute band weights:

For each band i, for each wavelength bin j, weight_ij = width_j if wavelength bin center falls inside band else 0. Width_j = wavelengths[j+1]-wavelengths[j] or difference between consecutive edges. We'll approximate width by difference between next wavelength minus current. For last bin, use same width as previous.

Simpler: We'll compute bin edges: wavelengths edges = np.concatenate([wavelengths[:1]-diff/2, (wavelengths[:-1]+wavelengths[1:])/2, wavelengths[-1]+diff/2]) but easier: use diff = np.diff(wavelengths); midpoints? Wait we have wavelength vector with equal spacing maybe. We can assume uniform spacing for simplicity. We'll set wavelengths = np.linspace(350, 750, 400) -> 400 points. Step = 1.0 nm. Then bin width 1 nm. Then weight_ij = 1 if wl in band else 0. So A is binary matrix. Then reconstruction solves linear system.

Ok.

Let's implement.

Also we might add small regularization to avoid singularities: use ridge regression or linear regression with fit_intercept False.

But we can just use np.linalg.lstsq.

Ok.

We will produce final code accordingly. Let's write.assistantfinal code