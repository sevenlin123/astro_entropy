import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis spectra
def create_basis_spectra(n_basis=3, n_points=100):
    wave = np.linspace(4000, 8000, n_points)  # Angstrom
    basis = []
    for i in range(n_basis):
        # Gaussian bump at different centers
        center = 4000 + i * 2000 / (n_basis - 1)
        width = 300
        spec = np.exp(-0.5 * ((wave - center) / width)**2)
        basis.append(spec)
    return wave, np.array(basis)

# 2. generate synthetic spectra
def synthesize_spectra(wave, basis, n_objects=50):
    coeffs = np.random.rand(n_objects, basis.shape[0])  # random weights
    spectra = coeffs @ basis   # linear combination
    noise = np.random.normal(scale=0.02, size=spectra.shape)
    spectra += noise
    return spectra, coeffs

# 3. generate photometric data from synthetic spectra
def filter_response(wave, filt_center, filt_width=200):
    return np.exp(-0.5 * ((wave - filt_center) / filt_width)**2)

def integrate_flux(spectrum, wave, response):
    return np.trapz(spectrum * response, wave)

def photometry_from_spectra(wave, spectra, filt_centers=[4500, 5500, 7000]):
    fluxes = []
    for filt in filt_centers:
        resp = filter_response(wave, filt)
        flux = integrate_flux(spectrum=spectrum,
                              wave=wave,
                              response=resp)
        fluxes.append(flux)
        if isinstance(spectra, np.ndarray):
            pass  # placeholder to keep the loop
    # handle multiple objects
    if len(spectra.shape) == 2:
        fluxes = []
        for i in range(spectra.shape[0]):
            f = []
            for filt in filt_centium? Wait?? We need...
We need to produce final code with proper functions. Let's craft again.

We need:

- function create_basis_spectra
- function synthesize_spectra
- function filter_response
- function integrate_flux
- function photometry_from_spectra
- function reconstruct_spectrum_from_photometry
- main block generating data and reconstructing

Let's implement photometry_from_spectra for all spectra. Should return array shape (n_objects, n_filters).

Also reconstruct using ridge regression: given photometric fluxes and basis integrated through filters.

Implement:

- compute filter responses once
- For each filter, compute integrated basis responses: dot of basis with filter integrals -> matrix of shape (n_objects? Actually basis integrated flux for each filter: shape (n_basis, n_filters)).
- Use ridge regression to fit coefficients: y (photometry) shape (n_objects, n_filters), X = basis_integrated_flux (n_basis, n_filters). But we need to solve for coeffs such that coeffs @ X ~ y. Equivalent to solving for coeffs for each filter separately: X^T X coeff = X^T y. Use ridge.

Implement: ridge_regression per filter: use Ridge(alpha=1.0, fit_intercept=False). Fit on X.T (n_filters, n_basis)? Wait: For each object, we have y_i (flux vector). We want to find coeffs_i such that coeffs_i @ B ≈ y_i where B is (n_basis, n_filters). So we can transpose: we can solve X = B.T shape (n_filters, n_basis). Then y_i shape (n_filters). Fit Ridge with X and y_i. Since we have many objects, we can fit using RidgeCV across all? For simplicity, we do solve via linear algebra: For each filter j, we have y[:, j] = coeffs @ B[:, j]. That's standard linear regression: y_j = B_j^T coeffs. So we can compute pseudo-inverse: coeffs = (B_j^T)^+ y_j. But for all j simultaneously, we can compute: coeffs = y @ np.linalg.pinv(B_j). However we must ensure shape.

Simpler: Use ridge on each filter: For each filter j, fit Ridge to predict y[:, j] from basis coefficients? Wait we don't have basis coefficients. Hmm.

We have unknown coefficients for each object. For each filter j, we have observed flux y[:, j] and known basis integrated flux B_j (vector length n_basis). Relationship: y[:, j] = coeffs @ B_j. Here coeffs shape (n_objects, n_basis). So this is linear system: y_j = coeffs * B_j. This is like linear regression where features are basis contributions B_j, but we want to solve for coeffs. We can treat B_j as design matrix with columns basis contributions? Actually each coefficient multiplies B_j scalar; thus for each object i: y_i_j = sum_k coeffs_i_k * B_j_k. This is linear in coeffs_i. So we can compute coeffs via ridge regression by setting X = B_j (1D vector) and solving for each filter separately? Wait we can't because we need n_objects outputs with same design.

Better: We can combine all filters into single linear system:

For all objects i and filters j: y_i_j = sum_k coeffs_i_k * B_j_k.

Define for each filter j a row vector B_j of length n_basis. The unknowns are coeffs_i_k for each i,k. We could solve for each object individually: For each i, we have vector y_i (length n_filters). Solve for coeffs_i by minimizing ||B * coeffs_i - y_i||^2 with Ridge. That's linear least squares with penalty. Use Ridge's solve function.

Thus we implement for each i: use Ridge(alpha=1.0, fit_intercept=False). Fit on X=B.T? Wait we want to predict y_i from B, but we have features = B.T? Let's think: B has shape (n_basis, n_filters). For each filter j, B[:,j] is vector of basis contributions. For each i, we want coeffs_i such that B^T @ coeffs_i = y_i. Because B^T shape (n_filters, n_basis). Multiply B^T (n_filters,n_basis) by coeffs_i (n_basis,) -> y_i (n_filters,). Yes! So we set design matrix X = B^T. Then for each i, y_i (n_filters). Use Ridge.fit(X, y_i). Since X same for all objects, we fit once? But we need separate coefficients for each i. So we can use Ridge regression by solving linear system: coeffs_i = (X^T X + alpha I)^-1 X^T y_i. That can be computed quickly using linear algebra.

Implementation:

X = B.T  # shape (n_filters, n_basis)
XT_X = X.T @ X
pinv = np.linalg.inv(XT_X + alpha*np.eye(n_basis))
coeffs = (y @ X) @ pinv? Wait let's compute properly:

For each i: coeffs_i = (X^T X + alpha I)^-1 X^T y_i.

Let A = (X^T X + alpha I)^-1 X^T. Then coeffs_i = A @ y_i. Since y_i shape (n_filters,), A shape (n_basis, n_filters). So compute A once: A = np.linalg.inv(XT_X + alpha*I) @ X.T. Then coeffs = y @ A.T? Wait y shape (n_objects, n_filters). Multiply y (n_objects, n_filters) by A.T (n_filters, n_basis) -> (n_objects, n_basis). Good.

So implement.

Then reconstruct spectra: coeffs @ basis.

Ok.

Let's write full script with these functions and a main demonstration.

Need to include synthetic data generation.

Also need to adjust photometry_from_spectra to return array.

Let's craft final code.assistantfinal code