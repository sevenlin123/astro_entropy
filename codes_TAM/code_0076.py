import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple linear spectral model
def linear_spectral_model(wavelengths, params):
    """
    Generate a synthetic spectrum as a weighted sum of basis functions.
    wavelengths : array-like, shape (n_wave,)
    params : array-like, shape (n_basis,)
    """
    # Basis: constant + slope + Gaussian bump
    const = np.ones_like(wavelengths)
    slope = wavelengths - wavelengths.mean()
    gauss = np.exp(-0.5 * ((wavelengths - 5000.) / 200.)**2)
    basis = np.vstack([const, slope, gauss])
    return basis.T @ params

# 2. generate synthetic spectra
def generate_synthetic_spectra(n_spec, n_wave, seed=0):
    """Create n_spec random spectra using the linear model."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(4000, 7000, n_wave)
    n_basis = 3
    true_params = rng.normal(size=(n_spec, n_basis))
    spectra = np.empty((n_spec, n_wave))
    for i in range(n_spec):
        spectra[i] = linear_spectral_model(wavelengths, true_params[i])
    return wavelengths, spectra, true_params

# 3. generate photometric data from synthetic spectra
def photometric_from_spectrum(spectra, wavelengths, filters):
    """
    Integrate each spectrum over each filter transmission curve.
    filters: list of tuples (lambda_start, lambda_end).
    """
    phot_vals = np.zeros((spectra.shape[0], len(filters)))
    for idx, (fstart, fend) in enumerate(filters):
        mask = (wavelengths >= fstart) & (wavelengths <= fend)
        # trapezoidal integration
        integ = np.trapz(spectra[:, mask], wavelengths[mask], axis=1)
        phot_vals[:, idx] = integ
    # normalize by filter width to get average flux
    widths = np.array([fend - fstart for fstart, fend in filters])
    phot_vals /= widths
    return phot_vals

# 4. reconstruct a synthetic spectrum from photometry
def reconstructions_from_photometry(phot, wavelengths, filters, n_basis=3):
    """
    Use ridge regression to estimate spectral parameters from photometry.
    The basis functions are evaluated on a dense wavelength grid.
    1) Build design matrix for photometry (M x P)
    2..P..A
    """
    # create design matrix A (for photometric observations)
    n_filters = len(filters)
    A = np.zeros((n_filters, n_basis))
    for i, (fstart, fint) = enumerate(filters):
        mask = (wavelengths >= fint[1]) ???????...
The assistant truncated. Let's craft full code. Need design matrix: For each filter, compute integral of basis function over that filter divided by width (average). So we can precompute basis integrated. Then solve ridge: params = ridge.fit(A, phot).predict? Actually we have phot observations and want to find params. We can fit ridge regression directly: param_est = ridge.fit(A, phot). But we need invert? Usually we fit: phot = A @ params + noise. So use Ridge regression with zero intercept. So: ridge = Ridge(alpha=1e-2, fit_intercept=False).params = ridge.fit(A, phot).coef_.transposed? Wait: For multiple target variables? We have phot shape (n_spec, n_filters). Want params for each spectrum: we fit ridge for each spec separately? We can fit linear regression per spec? But we can use RidgeCV but no. Simplify: use least squares: params_est = np.linalg.lstsq(A.T @ A, A.T @ phot.T) -> shape (n_basis, n_spec). That will give params. Then reconstruct spectra: evaluate linear_spectral_model with these params.

Implement:

def reconstruct_spectra_from_photometry(phot, wavelengths, filters):

Compute A: shape (n_filters, n_basis). Use same basis as model. For each filter compute mean integral.

Define helper integrate_basis(basis_func): compute integral.

Let's code.

Also generate synthetic data, run through.

Return code.assistantfinal code