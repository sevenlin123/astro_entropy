import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1) Define spectral model: simple linear combination of basis spectra
def define_spectral_basis(num_bases=5, num_wavelengths=1000):
    """
    Creates num_bases synthetic basis spectra over a wavelength grid.
    Returns wavelengths and basis matrix (num_wavelengths x num_bases).
    """
    wavelengths = np.linspace(400, 2500, num_wavelengths)  # nm
    # Gaussian basis spectra with random centers & widths
    rng = np.random.default_rng(seed=42)
    bases = []
    for _ in range(num_bases):
        center = rng.uniform(500, 2000)
        width = rng.uniform(50, 150)
        amp = rng.uniform(0.5, 1.5)
        spectrum = amp * np.exp(-0.5 * ((wavelengths - center)/width)**2)
        bases.append(spectrum)
    return wavelengths, np.column_stack(bases)

# 2) Generate synthetic spectra (linear combos of basis)
def generate_synthetic_spectra(n_samples=20, basis=None):
    """
    Generates n_samples synthetic spectra using random coefficients.
    Returns synthetic spectra matrix (n_samples x num_wavelengths).

The user wants minimal python implementation of core reconstruction framework ignoring validation methods, no plotting. Provide synthetic data generation.

We need to write code that defines spectral model, generates synthetic spectra, generates photometric data from synthetic spectra, and reconstructs synthetic spectrum from photometry.

Use numpy, scipy, sklearn.

We'll create basis spectra, generate synthetic spectra (combination). Then generate synthetic photometric data by integrating spectra over filter transmission curves. Use some synthetic filters: maybe 3 broad bands (g,r,i) with certain passbands.

Then reconstruction: given photometric colors, reconstruct spectrum via ridge regression or other method. We can train a model mapping photometric fluxes to spectra: i.e., treat each wavelength bin as target; we have training set of synthetic spectra and corresponding photometric measurements; train ridge regression model for each wavelength bin? That could be heavy but small dataset fine.

Simpler: Use linear inversion: solve for coefficients of basis given photometric measurements, then reconstruct spectrum. But that's basically same as generating synthetic spectra. But we can reconstruct by performing least-squares fit of basis coefficients using photometric integration.

Let's design: Photometric measurement m_k = integral S(λ)*T_k(λ) dλ. For each filter k we compute integral for each basis spectrum: B_{jk} = ∫ basis_j * T_k dλ. Then for synthetic spectrum s = sum_i a_i basis_i, photometry p_k = sum_i a_i B_{ik}. So we can recover a_i via linear regression (ridge) from photometry: p = B^T a -> a = (B^T)^+ p. So we can reconstruct spectrum.

Implement:

- define basis spectra
- generate synthetic spectra by random coefficients
- define filters (gaussian or rectangular)
- compute photometric vector per sample
- Train ridge regression: features = photometry, targets = basis coefficients? Actually we already have basis coefficients. For reconstruction demonstration, we can recover coefficients via ridge on training set and test.

But they said ignore validation methods; just demonstrate. So produce photometric data for all samples; reconstruct spectra using ridge: Fit ridge regression with photometry as X, spectra as y. Then predict spectra for each photometry.

Ok let's implement: