import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def create_spectral_model(wavelengths, n_components=5):
    """
    Construct a simple linear spectral model composed of Gaussian basis functions.
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_components)
    widths = (wavelengths.max() - wavelengths.min()) / (n_components * 4)
    bases = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        bases.append(g)
    return np.column_stack(bases)

def generate_synthetic_spectra(model, n_spectra=50, noise_std=0.01):
    """
    generate random coefficients and apply model to produce spectra.
    """
    coeffs = np.random.randn(n_spectra, model.shape[1])
    spectra = coeffs @ model.T
    # add noise
    spectra += np.random.normal(scale=noise_std, mean=0.0, size=spectra.shape)
    spectra = np.abs(np.clip(flux=???, min=..???)?? 
Scrolling: The user wants code with functions: define spectral model, generate synthetic spectra, generate photometric data from synthetic spectra, reconstruct synthetic spectrum from photometry. Use numpy, scipy, sklearn. Provide clear functions.

We need to produce runnable code. We must avoid syntax errors. Provide synthetic example input data inside script.

Let's design:

- define wavelengths array e.g., 400-800 nm step 5 nm.

- create_spectral_model: returns matrix of basis functions.

- generate_synthetic_spectra: given model, produce random coefficients, compute spectra.

- generate_photometric_data: given spectra and filter transmissions (e.g., U,B,V,R,I filters approximated), integrate: flux = integral(spectrum * filter)/integral(filter). We'll create simple filter transmission functions defined on same wavelengths.

- reconstruct_spectrum_from_photometry: treat photometric fluxes as linear combination of basis functions integrated over filters. Use linear regression to estimate coefficients that best reproduce photometric fluxes. Then reconstruct spectrum.

Simplify: For each filter j, we can compute vector pj = integrate(basis_i * filter_j) for all i. That gives matrix P (filters x basis). Solve for coefficients via Ridge regression using observed photometric fluxes.

Thus reconstruct_spectrum_from_photometry will:

- compute basis integration vectors for each filter
- use ridge regression to fit coefficients
- return reconstructed spectrum = coeffs @ model.T

We'll generate synthetic photometric data by integrating true spectra across filters.

Implementation steps:

1. Define wavelengths array: np.arange(400, 801, 5).

2. Define spectral model: using Gaussian basis functions across wavelengths.

3. Generate synthetic spectra: random coefficients, compute fluxes.

4. Define filter transmissions: U (350-420 nm), B (420-520 nm), V (520-600 nm), R (600-700 nm), I (700-800 nm). We'll approximate as top-hat.

- Filter trans function: array of ones where wavelength in range else zero.

5. Generate photometric data: for each spectrum, compute flux per filter: integrate(spectrum * filter) / integrate(filter) ; we'll use trapezoidal integration np.trapz.

6. Reconstruct: Using the same model, compute filter integrals for basis functions; build matrix P; use Ridge regression with photometric flux as target; compute coefficients; reconstruct spectra.

7. Compare original vs reconstructed spectra: plot? Not needed; but maybe compute residual RMS.

But no plotting required. We may just print RMSE.

Also need to ensure script runs: include main guard if __name__ == "__main__": and run demo.

Ok.

Edgecases: use numpy operations.

Now produce code accordingly. Ensure no placeholder "???" left.

Let's write functions:

- create_spectral_model(wavelengths, n_basis=10): returns model matrix shape (len(wavelengths), n_basis)

- generate_synthetic_spectra(model, n_spectra=20, noise=0.05): returns spectra array shape (n_spectra, len(wavelengths))

- make_filters(wavelengths): returns dict of filter name to transmission array.

- generate_photometry(spectra, filters, wavelengths): returns photometric flux array shape (n_spectra, n_filters). Use np.trapz.

- reconstruct_spectra_from_photometry(photometry, filters, wavelengths, model, alpha=1.0): returns reconstructed spectra.

We need to compute basis filter integrals: For each filter f, vector pf of length n_basis: pf[i] = integral(model[:,i] * filter)/integral(filter). So P matrix shape (n_filters, n_basis). Then we fit Ridge regression with alpha param.

Potential: we want to solve for each spectrum individually; but could solve simultaneously: For each spectrum, we have y vector of fluxes length n_filters; we need coeffs vector length n_basis; we use Ridge to find coeffs that minimize sum of squared difference across all spectra; we can fit individually using ridge for each spectrum: model.fit(P, y). But P is same for all; so we could do linear regression with n_samples=n_spectra and features=n_filters? Wait: Actually we want to fit each spectrum's coefficients separately. So for each spectrum we fit ridge: ridge.fit(P, y). That's fine.

Better: Use sklearn.linear_model.Ridge, set fit_intercept=False, solver='auto'.

We also might use sklearn.linear_model.LinearRegression with regularization? But Ridge is okay.

Define reconstruct_spectra_from_photometry: loops over each spectrum, fits ridge, obtains coefficients, compute recon_spectrum = coeffs @ model.T.

Return array.

Testing: In main, generate synthetic spectra, compute photometry, reconstruct, compute RMSE between true and reconstructed.

Print RMSE.

All good.

Make sure imports: numpy, scipy.integrate maybe not needed; sklearn.linear_model import Ridge.

Let's produce final code. Also we need to handle random seed for reproducibility: np.random.seed(42).

Ok. Let's craft final python script.

Check syntax: in generate_photometry, we use filter transmissions dict. Implementation:

def generate_photometry(spectra, filters, wavelengths):
    n_spectra = spectra.shape[0]
    n_filters = len(filters)
    phot = np.zeros((n_spectra, n_filters))
    for idx, (name, trans) in enumerate(filters.items()):
        # integration
        denom = np.trapz(trans, wavelengths)
        for i in range(n_spectra):
            num = np.trapz(spectra[i] * trans, wavelengths)
            phot[i, idx] = num / denom
    return phot

This works but loops. Acceptable.

Similarly, in reconstruct_spectra_from_photometry:

- Build P matrix: for each filter: pf vector = integral(model[:,i]*trans)/denom for each basis i.

Implement:

def compute_filter_basis_integrals(model, filters, wavelengths):
    n_filters = len(filters)
    n_basis = model.shape[1]
    P = np.zeros((n_filters, n_basis))
    for j, (name, trans) in enumerate(filters.items()):
        denom = np.trapz(trans, wavelengths)
        for k in range(n_basis):
            num = np.trapz(model[:,k] * trans, wavelengths)
            P[j, k] = num / denom
    return P

Then reconstruct: loop over spectra.

ridge = Ridge(alpha=alpha, fit_intercept=False)

coeffs = ridge.fit(P, phot[i]).coef_

recon_spec = coeffs @ model.T

Collect.

Ok.

Need to ensure arrays shapes: P shape (n_filters, n_basis). Ridge expects X shape (n_samples, n_features). Here we treat each spectrum as one sample? Actually we are solving for coefficients that map basis functions to filter measurements. The mapping is: phot[i,:] ≈ P @ coeffs_i. P shape (n_filters, n_basis), we need to solve linear system: P^T * coeffs_i ≈ phot[i,:]. Wait orientation: Suppose basis_i (length n_wavelength). For each filter j, flux_j = ∫ S(λ) T_j(λ) dλ / ∫ T_j dλ. S = Σ coeffs_k * basis_k. So flux_j = Σ coeffs_k * ∫ basis_k T_j / denom_j = Σ coeffs_k * P_jk. So flux vector y = P * coeffs. Where P shape (n_filters, n_basis). So y = P @ coeffs. So we need to solve for coeffs given y and P.

Ridge can be used to solve y = P @ coeffs: we set X=P, y=y, ridge.fit(X,y). coefficient vector returned by ridge.coef_. Works.

So correct orientation: ridge.fit(P, phot[i]) yields coeffs. Good.

All good.

Let's produce final code. Ensure no extraneous prints except final RMSE.

Add demonstration of multiple spectra.

Ok. Let's produce final answer.assistantfinal code