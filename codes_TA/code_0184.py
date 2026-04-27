import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis spectra
def spectral_model(params, wavelengths):
    """
    params: array-like, coefficients for each basis spectrum
    wavelengths: array, wavelength grid
    Returns: modeled spectrum (array)
    """
    # Define two simple basis spectra: Gaussian peaks
    basis1 = np.exp(-0.5 * ((wavelengths - 500) / 30)**2)
    basis2 = np.exp(-0.5 * ((wavelengths - 700) / 50)**2)
    return params[0] * basis1 + params[1] * basis2

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths):
    """
    n_samples: number of synthetic spectra to generate
      wavelengths: array of wavelength points
    returns: array shape (n_samples, len(wavelengths))
    """
    spectra = []
    for _ in range(n_samples):
        coeffs = np.random.uniform(0.5, 1.5, size=2)
        spectrum = spectral_model(coefficients=coeffs,
                                  **kwargs??? ?????? ??????, 
We need to provide full code that generates synthetic spectra, photometric data, and reconstructs a synthetic spectrum from photometric data using some inverse method (ridge regression). The user wants minimal but functional code.

Let's design:

- wavelengths grid: e.g., 400-800 nm with steps 5 nm.

- Basis spectra: e.g., Gaussian peaks at 520 and 660 nm.

- Synthetic spectra generation: random coefficients times basis spectra plus noise.

- Photometric data: defined filter response curves (e.g., 3 broad filters centered at 450, 600, 750 nm). For each spectrum, compute integrated flux across each filter: integral of S(λ)*R(λ)dλ normalized by integral of R(λ)dλ.

- Reconstruction: given photometric vector p (size m), reconstruct spectrum via ridge regression using filter responses matrix (m x len(wavelengths)). Solve for weights in spectral basis? But we can reconstruct directly by fitting spectral basis coefficients from photometry.

Simplest: assume the true underlying basis are known (the two Gaussians). We have photometric data p. We want to recover coefficients c such that basis1*c1 + basis2*c2 approximates underlying spectrum. Use linear regression: we know photometric response to each basis: r_i_k = integral(basis_k * filter_i). Then photometry vector p ~ R * c where R is matrix m x 2. Solve via ridge regression.

Alternatively, use full wavelength grid: We can reconstruct spectrum directly by solving linear system: p = T * s, where T is matrix m x N (filter responses integrated over wavelengths). But this is underdetermined; we use regularization (ridge) to obtain s.

Let's implement:

- Build filters: gaussian response around centers 450, 600, 750 nm with width 30 nm.

- Compute filter_response matrix: For each filter i, integrate basis? Actually we need filter matrix: for each filter, compute filter value at each wavelength.

- To reconstruct spectrum s: solve minimize ||T @ s - p||^2 + alpha||s||^2. Use Ridge with alpha=1.0, fit intercept=False.

Implementation:

- generate_synthetic_spectra: produce many spectra, store coefficients used.

- generate_photometry: for each spectrum, compute photometric values.

- reconstruct_spectrum: given photometry vector p, use ridge regression on matrix T (filters over wavelengths) to get s. Optionally use precomputed coefficients.

But need to make script runnable; maybe generate one synthetic spectrum, compute photometry, reconstruct, plot difference? But plotting is optional; still we can just print.

The code should be minimal but functional.

Let's implement:

Functions:

- spectral_basis(wavelengths): returns basis matrix 2 x N.

- synthetic_spectrum(coeffs, wavelengths): return coeffs[0]*basis1 + coeffs[1]*basis2.

- generate_filters(wavelengths): return filters matrix m x N.

- photometry(spectrum, filters): compute dot product of spectrum with each filter (integrate approximate sum). Use numpy.trapz.

- reconstruct_from_photometry(p, filters, alpha=0.01): solve ridge regression: s_hat = ridge.fit(filters.T, p). Actually ridge expects X: n_samples x n_features. So filters.T is N x m? Wait.

We have T (m x N). We want to find s (N,) minimizing ||T*s - p||^2 + alpha||s||^2. This is standard ridge: s = (T^T T + alpha I)^(-1) T^T p. Use numpy.linalg.solve.

Implement using numpy.linalg.lstsq with regularization? Let's just compute manually.

Compute A = T.T @ T + alpha * np.eye(N); b = T.T @ p; s = np.linalg.solve(A, b).

Return s.

Now main script: create wavelengths, basis, filters. Generate synthetic coefficients random; create true spectrum. Compute photometry. Reconstruct spectrum. Print RMSE between true and reconstructed.

We don't need cross-validation etc.

Let's write code accordingly.

Also must be careful with function signatures. Provide minimal docstrings optional.

Edge: generate_synthetic_spectra may produce list of spectra and corresponding coeffs.

We'll generate one sample to keep simple.

Also we can generate multiple for demonstration.

Ok final code.

Let's produce final python script.

Note: Use numpy, scipy (maybe not necessary). scikit-learn not needed if we implement ridge manually; but requirement includes scikit-learn, so we can use Ridge from sklearn.linear_model to demonstrate.

We can use Ridge: X = filters.T (N x m), y = p. But Ridge expects X shape (n_samples x n_features). Here we have 1 sample? Actually we want to predict spectrum vector of length N from photometry vector p (size m). But that's not typical usage of Ridge: ridge is supervised learning: mapping inputs to target outputs. We have training data: we could create training set of synthetic spectra and their photometry. Fit Ridge to learn mapping from photometry to spectrum. Then apply to test photometry.

Thus we can implement:

- Generate many synthetic spectra and photometry pairs.

- Train Ridge regressor on photometry (m features) to predict spectrum (N targets). Use Ridge with multioutput support: y shape (n_samples, N). We can fit.

- Then reconstruct on new photometry.

Let's do that: simpler.

Process:

1. Create basis, filters.

2. Generate dataset: n_train=1000 spectra: coeffs random, compute spectra, photometry.

3. Train Ridge(alpha=1.0) on X=p, y=spectra.

4. Generate a test spectrum and photometry, predict spectrum.

5. Compute RMSE.

That uses scikit-learn as requested.

Ok implement.

Note: we need to generate synthetic photometry: integrate using trapz of spectrum*filter over wavelengths divided by integral of filter.

Define functions accordingly.

Return code.

Let's produce final answer.assistantfinal code