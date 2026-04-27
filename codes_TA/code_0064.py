import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------------
# 1) Spectral model ----------------------------------------------------
# --------------------------------------------------------------------
def gaussian_spectrum(wl, amp, cen, sigma):
    """Single Gaussian line."""
    return amp * np.exp(-(wl - cen)**2 / (2 * sigma**2))

def composite_spectrum(wl, params):
    """
    Sum of Gaussians.
    params : list of (amp, cen, sigma) tuples
    """
    spec = np.zeros_like(wl)
    for amp, cen, sigma in params:
        spec += gaussian_spectrum(wl, amp, cen, sigma)
    return spec

# --------------------------------------------------------------------
# 2) Synthetic spectra ------------------------------------------------
# --------------------------------------------------------------------
def generate_synthetic_spectra(n_spec, wl_grid, params_list):
    """
    Create n_spec synthetic spectra using parameters from params_list.
    Each entry of params_list is a set of (amp,cen,sigma) tuples.
    """
    spectra = []
    for params in params_list[:n_spec]:
        spectra.append(composite_spectrum(wl_grid, params))
    return np.array(spectra)

# --------------------------------------------------------------------
# 3) Photometry from spectra ------------------------------------------
# --------------------------------------------------------------------
def integrate_band(spectrum, wl, band_center, band_width):
    """Integrate spectrum over a Gaussian filter centred at band_center."""
    filt = np.exp(-((wl - band_center)**2) / (2 * band_width**2))
    return simps(spectrum * filt, wl)

def photometric_integration(spectra, wl, band_centers, band_widths):
    """Compute fluxes for each spectrum in given bands."""
    nspec = spectra.shape[0]
    fluxes = np.empty((nspec, len(band_centers)))
    for i, (c, w) in enumerate(zip(band_centers, band_widths)):
        filt_flux = np.array([integrate_band(sp, wl, c, w) for sp in spectra])
        fluxes[:, i] = filt_flux
    return fluxes

# --------------------------------------------------------------------
# 4) Spectrum reconstruction ------------------------------------------
# --------------------------------------------------------------------
def reconstruct_spectrum(fluxes, wl, band_centers, band_widths,
                         n_components=10, alpha=1.0):
    """
    Reconstruct a spectrum from broadband fluxes using linear ridge
    regression on a basis of Gaussian trial spectra.
    """
    # Build design matrix from Gaussian basis functions
    X = []
    for amp in np.linspace(0.5, 2.0, n_components):
        basis_sps = [gaussian_spectrum(wl, amp, cen, sigma)
                     for cen, sigma in zip(band_centers, band_widths)]
        for sp in basis_sps:
            X.append(simps(sp, wl))  # integral of basis? Actually flux
    X = np.array(X).reshape(-1, len(band_centers))
    # Create training data: X -> fluxes
    reg = Ridge(alpha=alpha)
    reg.fit(X.T, fluxes.T)
    # Predict coefficients for new target fluxes (here we use same)
    coeffs = reg.predict(X.T)
    # Construct reconstructed spectrum by summing weighted basis spectra
    recon = np.zeros_like(wl)
    idx = 0
for amp in range(1, n_components+???) :
    pass

def main():
    # Define wavelength grid
    wl = np.linspace(3000, 10000, 1000)
    # parameters for two synthetic spectra
    params_list = [
        [(1.0, 4000, 200), (0.5, 6000, 300)],
        [(0.8, 4500, 250), (0.6, 6000, 120)]
    ]
    # generate spectra
    spectra = generate_synthetic_spectra(2, wl, params_list)
    # photometric bands
    band_centers = [3800, 5200, 6500, 7500]
    band_widths = [200, 150, 200, 200]
    # derive fluxes
    fluxes = photometric_integration(spectra, wl, band_centes????

We need to produce working code that satisfies requirements. The previous attempt had errors. Let's design again.

Need functions:

- spectral model: maybe just simple base lines, but we can use arbitrary shape.

- generate synthetic spectra: use composite spectrum.

- photometry: integrate over filters.

- reconstruction: we can treat reconstruction as solving linear least squares between measured band integrals and spectral basis (e.g., basis set of sample points). Approach: create matrix A where rows are band responses to each wavelength point, columns correspond to wavelengths. For each band, compute response function over wl grid. Then flux vector f = A * s where s is unknown spectrum values. Solve s via linear regression or least squares. Use Ridge regression.

Simpler: A is shape (n_bands, n_wavelengths). We have flux vector f (n_bands). We want to reconstruct s (n_wavelengths). Solve via Ridge: s = (A^T A + alpha I)^-1 A^T f.

Implement function reconstruct_spectrum_from_photometry(fluxes, wl, band_centers, band_widths, alpha).

Will produce reconstructed spectra for each flux vector.

We will generate synthetic spectra, compute fluxes, then reconstruct using same fluxes, compare with original.

Let's implement.

Edge cases: Need to create filter responses: Gaussian filters.

Define integrate_band: integrate spectrum times filter.

But for building A, we can compute each band's filter over wl. Then A[i,j] = filter_i[j]. So flux f_i = sum_j filter_i[j] * spectrum[j] * delta_wl approximated by integration. But we used simps earlier. For simplicity, we can approximate flux = sum_j filter_i[j]*spectrum[j] * dlambda. Equivalent to dot product. We can incorporate dlambda into filter.

We'll compute A as filter vectors multiplied by delta_lambda. delta_lambda = wl[1]-wl[0] constant.

Thus f = A @ s.

Reconstruction: solve s.

Implement.

Also we may add small noise to fluxes.

Let's produce final code.

Be careful with variable names.

Let's write code accordingly.

We'll also include main() that prints MSE.

Let's produce final code.assistantfinal code