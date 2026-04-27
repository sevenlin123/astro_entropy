import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Define spectral model (linear combination of basis spectra)
# ------------------------------------------------------------------
def construct_basis(num_bases, num_points, rng):
    """Generate random basis spectra over wavelength grid."""
    return rng.standard_normal((num_bases, num_points))

def linear_spectrum(coeffs, basis):
    """Compute spectrum as linear combination of basis spectra."""
    return coeffs @ basis

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(num_objects, num_bases, num_points, rng):
    """
    Generate synthetic spectra with random coefficients.
    Each spectrum is linear combination of basis spectra.
    """
    basis = construct_basis(num_bases, num_points, rng)
    coeffs = rng.uniform(-1, 1, size=(num_objects, num_basis))
    return basis, coeffs, linear_spectrum(coeffs, basis)

# ------------------------------------------------------------------
# 3. Photometry from synthetic spectra
# ------------------------------------------------------------------
def photometry_from_spectrum(spectrum, wave, filters):
    """
    Compute photometric fluxes by integrating spectrum across filter
    transmission curves.
    - spectrum: (N, P) where N is #objects, P is # wavelengths.
    - **filters**: list of tuples (wave_filt, trans_filt).
    Returns fluxes (N, F), where F is number of filters.
    """
    fluxes = []
    for wf, tf in filters:
        interp_f = interp1d(wave, *[spectrum[i] for i in range(spectrum.shape[0])], axis=1,
                            fill_value="extrapolate", kind='cubic')
        # This approach is inefficient but simplifies interpolation
        N = spectrum.shape[0]
        integrand = np.array([interp_f(wf)*tf(wf) for wf in wave])
            # ... 
        ??? 
We need to write code fulfilling specification. Must be runnable. The previous attempt had issues. Let's design functions.

We'll create wavelength grid, basis spectra, random coefficients, linear spectra. Then compute photometry: integrate product of spectrum and filter transmission over wavelength. We'll use simple trapezoidal integration.

Define filters: e.g., Gaussian transmissions centered at certain wavelengths.

Implement photometry_from_spectrum to return fluxes.

Then reconstruction: given photometric fluxes and filter transmissions, reconstruct spectrum via linear inverse problem using regularized least squares (Ridge). We'll build design matrix A where each row corresponds to integral of basis spectra through filter. Solve for coefficients.

Steps:

1. create basis spectra over wavelengths array wave.
2. For each basis spectrum j, compute photometric response matrix: integrate basis[j] * filter_transmission over wave -> produce vector of length F.
   So A shape (F, B) (filters rows, basis columns). For each object, we have coefficients c, and predicted fluxes f_pred = A^T? Wait: flux = integral(spectrum * filter). Spectrum = sum_j c_j * basis_j(w). So flux_i = sum_j c_j * integral(basis_j * filt_i). So flux = A * c where A_{i,j} = integral(basis_j * filt_i). So A shape (F,B).

3. We have observed fluxes f_obs (N,F). For each object, solve ridge regression: minimize ||A c - f||^2 + alpha * ||c||^2. Use Ridge solver for each object or solve via closed form.

Simplest: use Ridge regression from sklearn: fit to training data where X = A.T? Wait, typical usage X shape (n_samples, n_features). Here n_samples = N objects, n_features = B basis. We want to predict fluxes F from basis coefficients. But we need to map c->flux. Actually we can treat basis coefficients as unknown variables; we want to fit mapping from basis to fluxes. We can set up X as each object's basis representation: For each object, we have vector of basis spectra evaluated at all wavelengths. But we don't know that; we only have fluxes. However we can reconstruct coefficients via regression: treat fluxes as target, features as ??? Not good.

Alternative: Since we know basis and filter responses, we can invert matrix A. Use least squares: For each object, we have f_obs, unknown c. Solve A c = f_obs. Use Ridge to regularize: c = ridge_solver(A, f_obs). That uses normal equations: (A^T A + alpha I) c = A^T f. Compute solution.

So implement function reconstruct_coeffs(f_obs, A, alpha). Use numpy.linalg.solve.

Then reconstruct spectrum: sum over basis * c.

Let's produce synthetic data.

Implementation details:

- wave: linspace 300 nm to 800 nm, 500 points.
- num_bases = 5
- num_objects = 10
- rng = np.random.default_rng(seed=42)

Construct basis: random normal.

Coeffs: uniform -1 to 1.

Compute spectra.

Define filters: say 3 filters: Gaussian centers at 400, 550, 700 nm; widths 50 nm.

Define function gaussian_filter(wave, center, width): returns normalized Gaussian.

Compute photometry: For each filter, integrate spectrum * filter over wave using np.trapz.

Implement photometry_from_spectrum: loops over filters.

Then compute A: for each filter, compute integral of each basis spectrum times filter.

Reconstruct coefficients: ridge_regression.

Reconstruct spectra: use coefficients.

Plotting optional omitted.

Provide main function demonstrating.

Need to ensure functions defined properly.

Let's craft code.

Note: Use numpy arrays; careful shapes.

Implement gaussian_filter: returns array same shape as wave.

Implement generate_filters: returns list of filter transmission arrays.

Let's produce code.assistantfinal code