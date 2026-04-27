import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# ------------------------------------------------------------------
# 1) Spectral model
# ------------------------------------------------------------------
def spectral_model(wavelengths, temp, logg, feh):
    """
    Simple black‑body scaled by metallicity, with a linear gravity term.
    wavelengths : array of λ [nm]
    temp        : effective temperature [K]
    logg        : log10(g) [cgs]
    feh         : [Fe/H] metallicity [dex]
    Returns flux density at each wavelength.
    """
    # Planck function in arbitrary units
    h = 6.62607015e-34
    k = 1.380649e-23
    c = 299792458.0

    def planck(lam, T):
        lam_m = lam * 1e-9
        return (2*h*c**2) / (lam_m**5 * (np.exp(h*c/(lam_m*k*T)) - 1))

    flux = planck(wavelengths, temp)
    # scale with metallicity (simple multiplicative factor)
    flux *= 10**(feh)
    # linear gravity dependence
    flux *= (1 + 0.1*(logg - 4.5))
    return flux

# ------------------------------------------------------------------
# 2) Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths):
    rng = np.random.default_rng()
    temps = rng.uniform(4000, 8000, n_samples)
    loggs = rng.uniform(3.5, 5.0, n_samples)
    feas = rng.uniform(-1.5, 0.5, n_samples)

    spectra = np.vstack([spectral_model(wavelengths, t, g, f)
                         for t, g, f in zip(temps, loggs, feas)])
    # normalize
    spectra /= np.max(spectra, axis=1, keepdims=True)
    return spectra, temps, loggs, feas

# ------------------------------------------------------------------
# 3) Generate photometric data
# ------------------------------------------------------------------
def photometric_response(wavelengths, response_centers, response_widths):
    """Generate band response profiles."""
    responses = []
    for cen, wid in zip(response_centers, response_widths):
        resp = np.exp(-(wavelengths - cen)**2 / (2*wid**2))
        resp /= np.mean(resp)
    return np.array(responses)  # shape (n_bands, n_wavelengths)

def get_photometric_fluxes(spectrum, responses):
    # integrate flux * response over wavelengths
    integrals = np.trapz(spectrum[:, None] * responses, axis=-1)
    return integrals

def create_photometric_reconstruction(
        spectrum,  # [n_pixels?]
        photo_resp, 1? etc
We need to produce runnable code that includes all steps: define model, generate synthetic spectra, generate photometric data, reconstruct synthetic spectrum from photometric data. Provide clear functions. Use numpy, scipy, scikit-learn. Not show reasoning, just code.

I need to finish the code. The last part incomplete: we need to reconstruct the spectrum from photometric data using regression: given photometric band fluxes, estimate spectrum. This is ill-posed but we can do linear regression where each spectrum is represented by basis functions (e.g., polynomial basis or PCA). For simplicity, we can use ridge regression with basis derived from training set: compute basis as mean + principal components of spectra. Then for new photometric data, fit coefficients via ridge to reconstruct.

But we need to generate photometric data from synthetic spectra. Provide functions: generate photometric data from synthetic spectra: pass through band responses. Then reconstruct spectrum: given photometric fluxes, find best-fit coefficients to reproduce photometric fluxes by applying same basis. Approach: compute basis functions from training spectra. Fit photometric fluxes as linear combination of basis functions weighted by their integrated photometric responses. Solve for coefficients with ridge.

Simplify: Use linear regression on photometric fluxes to predict full spectrum: fit mapping from photometric vector to spectrum vector using linear ridge regression. That is, use training data: X_phot (n_samples x n_bands) -> Y_spec (n_samples x n_wavelengths). Use RidgeCV to fit coefficient matrix W (n_bands x n_wavelengths). Then reconstruct for new photometric vector: Y_pred = X_new @ W.

Let's implement: train ridge with cross-validation.

Also need to generate synthetic example input data within script: call main function: generate synthetic spectra, photometric responses, photometric data, train ridge, test.

Also not require plotting.

Define functions: build_band_responses, generate_photometry, train_ridge, reconstruct_spectrum.

Let's produce code accordingly.

Edge: shape: spectra shape (N, M). Photometry shape (N, B). Response shape (B, M).

Compute photometry: for each sample, compute integral of spectrum*response across wavelengths. Use trapz.

Train ridge: X = photometry; Y = spectra. Use RidgeCV with alphas.

Predict: new photometry -> spectra.

Will generate synthetic dataset: Ntrain=200, Ntest=20. Build band responses: e.g., 5 bands centered at 400, 500, 600, 700, 800 nm widths 50 nm.

We'll produce code accordingly.

Also ensure that the script runs when executed.

Let's write code.assistantfinal code