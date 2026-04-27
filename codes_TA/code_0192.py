import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define spectral model
def spectral_model(wavelengths, params):
    """Simple linear combination of Gaussian basis functions."""
    n_basis = len(params) // 2
    mu = params[:n_basis]
    sigma = params[n_basis:]
    flux = np.zeros_like(wavelengths)
    for m, s in zip(mu, sigma):
        flux += np.exp(-0.5 * ((wavelengths - m) / s)**2)
    return flux

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wave_start, wave_end, n_points, params_list=None):
    """
    Generates n_spectra synthetic spectra using random Gaussian parameters.
    Returns wavelengths and fluxes.
    """
    wavelengths = np.linspace(wave_start, wave_end, n_points)
    fluxes = []
    if params_list is None:
        params_list = []
        for _ in range(n_spectra):
            # Two Gaussian components per spectrum
            mu = np.random.uniform(4000, 8000, size=2)
            sigma = np.random.uniform(50, 150, size=2)
            params_list.append(np.concatenate([mu, sigma]))
    for params in params_list:
        flux = spectral_model(wavelengths, params)
        fluxes.append(flux)
    return wavelengths, np.array(fluxes)

# 3. Photometric data generation
def generate_photometry(spectra, wavelengths, filter_wavelengths, filter_transmissions):
    """
    Convolve spectra with filter transmission curves to get integrated fluxes.
    """
    phots = []
    for spec in spectra:
        integrals = []
        for filt_wl, filt_tr in zip(filter_wavelengths, filter_transmissions):
            f_interp = interp1d(wavelengths, spec, kind='linear', bounds_error=False, fill_value=0.0)
            integ = np.trapz(filt_tr * f_interp(filt_wl), filt_wl)
            integrals.append(integ)
                        # 
                    # 
                    # 
                    # 
                #
                #
                #  
        phots.append(integrals)
    return np.array(phots)

# 4. Reconstruction from photometry
def reconstruct_spectrum_from_photometry(photon_data, wavelengths, filter_wavelengths, filter_transitions):
    """
    Use Ridge regression to match photometric data to basis spectral templates.
    """
    # Create matrix A where each column is a spectrum in the basis set
    # (i.e., we are using our first set of synthetic spectra as reference).
    # ...
    # We'll create synthetic base spectra via random gaussian combos.
    base_spectra = []
    for i in range(20):
        mu = np.random.uniform(4000, 7200, self????)?   ** 

We must produce final code that implements algorithm: spectral model, generate synthetic spectra, photometry, reconstruction via linear regression. Use numpy, scipy, sklearn. Provide functions. Let's design.

We can implement simple basis: use same spectral_model as basis functions; but simpler: we generate a set of basis spectra by varying parameters. Then photometry: convolve with filter transmissions.

Reconstruction: given photometric data, we want to find coefficients to reconstruct a spectrum. We'll treat each basis spectrum as features in regression: We have photometry vector p (size n_filters). We want to solve for coefficients w such that M w ≈ p, where M_{ij} = integrated flux of basis spectrum j through filter i. Solve via ridge regression. Then reconstruct spectrum as sum_j w_j * basis_spectrum_j. Use training data: we might have a training set of synthetic spectra with known photometry to fit ridge regression? Actually we can directly compute M from basis spectra and solve for w. But we still need to know weights that minimize difference between observed photometry and predicted photometry of reconstructed spectrum. That's what ridge solves.

We'll generate synthetic training data: many synthetic spectra, each with known photometry. We can compute basis matrix M for all training spectra? That would be huge. Instead we can just use the same set of basis spectra to represent any spectrum; the weights found by ridge will approximate.

So steps:

1. generate N_train spectra using random params.
2. generate photometry for them.
3. Build matrix X (N_train x N_basis) where columns are photometry of each basis spectrum? Wait. For ridge we need design matrix of predictors (basis spectra) and target photometry? But we want to reconstruct a new spectrum given its photometry. Equivalent: treat each basis spectrum's photometry as a feature vector; given target photometry y, find weights w such that M^T w ≈ y? Let's think:

If we have M (n_filters x n_basis) as integrated flux for each basis through filters. For a given unknown spectrum represented as combination of basis spectra: flux = sum_j w_j * B_j (where B_j is basis spectrum). Its photometry y = M * w. So given y, we want w solving M*w = y (least squares). So we can solve directly via linear algebra: w = (M^T M + alpha I)^{-1} M^T y. That's ridge regression.

Thus we don't need training set; we just need basis spectra and their photometry. So step 4: compute M from basis spectra. Then for each photometry vector y, solve for w and reconstruct spectrum as weighted sum of basis spectra.

Let's implement functions accordingly.

We'll need filter definitions: we'll generate simple Gaussian filters. Eg filter transmissions as gaussian centered at certain wavelengths.

Implementation plan:

- spectral_model(wavelengths, mu, sigma): returns sum of Gaussians.
- generate_synthetic_spectra(n_spectra,...): returns wavelengths and fluxes.
- generate_filters(n_filters, wave_start, wave_end): returns list of filter wavelengths arrays and transmissions arrays. Use gaussian filter shapes.

- generate_photometry(spectra,wavelengths,filters): compute integrated flux per filter.

- reconstruct_spectrum_from_photometry(y, wavelengths, basis_fluxes, filters): build M (n_filters x n_basis). Solve ridge. Return reconstructed flux.

We'll also provide main function to test: generate basis spectra, compute photometry, pick one target spectrum, reconstruct it, compare.

Implement ridge using sklearn Ridge with fit_intercept=False.

Let's write code accordingly. Ensure self-contained.

Now produce final code.assistantfinal code