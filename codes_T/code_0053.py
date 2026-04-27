import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# Define wavelength grid for synthetic spectra (Angstroms)
lam_grid = np.linspace(4000, 7000, 300)  # 300 points from 4000 to 7000 Å

def spectral_model(params):
    """
    Simple Gaussian spectral model.
    params: array-like, [amplitude, center, width]
    Returns flux array on lam_grid.
    """
    amp, cen, wid = params
    return amp * np.exp(-0.5 * ((lam_grid - cen) / wid) ** 2)

def generate_synthetic_spectra(n_samples=10):
    """
    Generate synthetic spectra by sampling parameters.
    """
    np.random.seed(42)
    amps = np.random.uniform(0.5, 1.5, n_samples)
    cents = np.random.uniform(4500, 6500, n_samples)
    widths = np.random.uniform(50, 200, n_samples)
    spectra = []
    for a, c, w in zip(amps, cents, widths):
        spectra.append(spectral_model([a, c, w]))
    return np.array(spectra), np.vstack([amps, cents, widths]).T

def bandpass_filter(lambda_min, lambda_max, resolution=100):
    """
    Creates an arbitrary bandpass filter function.
    """
    lam_vals = np.linspace(lambda_min, lambda_min, resolution)
    # 2-pixel width at center
    center = (lambda_min + lambda_max)/ 2
    return (lambda_min <= lam_vals <= lambda_max)

def create_photometry(spectra, filters):
    """ 
    S takes input; uses rest-frame spectra. 
     
     
    This function ...  

-   ? ...
We need to provide python code implementing the requirements: define spectral model, generate synthetic spectra, generate photometric data from synthetic spectra, reconstruct a synthetic spectrum from photometric using ridge regression. Use numpy, scipy, scikit-learn. Provide clear functions. Minimal but functional.

Let's design: spectral model as simple gaussian. Synthetic spectra: sample amplitude, center, width. Photometry: compute flux through filter transmission curves. For simplicity we use top-hat filters with defined ranges. Use convolution integral: flux = sum(S * T) / sum(T). Then reconstruct: build basis matrix from spectra? Actually reconstruct a synthetic spectrum given photometry: Use ridge regression where design matrix is filter responses for basis spectra, target photometry. But we have limited number of basis spectra; we can reconstruct any spectrum as linear combination of basis spectra. We'll generate many synthetic basis spectra, build matrix of photometric fluxes for each basis spectrum. Then for a new photometry vector, solve ridge regression to find coefficients. Then reconstruct spectrum by weighted sum of basis spectra.

Implement functions:

- spectral_model(params)
- generate_synthetic_spectra(n_samples)
- create_filters(): returns list of filter transmission arrays
- compute_photometry(spectra, filters)
- reconstruct_spectrum(photon, basis_photometry, basis_spectra, alpha=1.0)

Add main block that generates data, splits into training/test, trains ridge, tests.

But requirement says ignore validation methods. So just do full training on all.

Also generate synthetic example input data inside script. So we will call functions.

Return only code. Let's craft.

Edge cases: bandpass: simple rectangular filters: defined by min, max. Use lam_grid. Filter transmisson array length same as lam_grid, 1 inside range else 0.

Compute photometry: integrate flux * T / integrate T -> average flux within band. Equivalent to dot product.

Reconstruction: Using Ridge regression: X_train = basis_photometry (N_basis x N_filters). y_train = basis_spectra (N_basis x N_flux_points). Ridge fit: For each wavelength point we can train separate ridge models, but easier: solve matrix problem via linear regression with regularization. However sklearn's Ridge reg expects shape (n_samples, n_features). We'll treat features as filters, target as spectral flux at each wavelength. We can fit one Ridge per wavelength: that may be heavy but fine. But simpler: use linear algebra: Solve (X^T X + alpha I)^{-1} X^T Y, where Y shape (N_filters, N_flux). Wait Y shape: each training sample has spectrum flux vector length N_flux. We want to predict flux vector as combination of filter responses. So we can compute coefficients for each wavelength: solve for B where X B = Y. Each column of Y corresponds to flux at one wavelength. Use ridge: B = inv(X^T X + alpha I) X^T Y. Then reconstruct: spec = ph * B. Where ph shape (1,N_filters). We'll implement.

Let's write functions:

- create_filters(): returns list of T arrays, maybe 5 filters across range.

- compute_photometry(spectra, filters): returns array shape (n_spectra, n_filters).

- reconstruct_spectrum(ph, basis_photometry, basis_spectra, alpha=1.0):

    compute B = inv(XTX+alpha I) X^T Y

    then reconstructed = ph @ B.

Return spec.

Add main test:

- generate basis spectra n=50

- compute basis photometry.

- generate new synthetic spectra n=5

- compute photometry.

- for each, reconstruct.

Print mean squared error between true spectrum and reconstructed.

Return code. Ensure imports: numpy, sklearn.linear_model.Ridge not used maybe.

But using numpy linear algebra directly is fine.

Add random seed for reproducibility.

Let's craft code accordingly.assistantfinal code