#!/usr/bin/env python3
import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

def generate_basis(n_wave, n_basis, rng=None):
    """
    Create a set of smooth basis spectra.
    """
    rng = rng or np.random.default_rng()
    # Randomly generate peaks positions and widths
    peaks = rng.uniform(0, n_wave, size=(n_basis, 3))  # center, width, amplitude
    x = np.arange(n_wave)
    basis = np.zeros((n_wave, n_basis))
    for i, (c, w, a) in enumerate(peaks):
        basis[:, i] = a * np.exp(-0.5 * ((x - c) / w)**2)
    return basis

def generate_synthetic_spectra(basis, n_samples, noise_std=0.01, rng=None):
    """
    Generate synthetic spectra as random linear combinations of basis spectra.
    """
    rng = rng or np.random.default_rng()
    n_basis = basis.shape[1]
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis.T  # shape (n_samples, n_wave)
    noise = rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

def generate_filter_responses(n_filters, n_wave, wavegrid, rng=None):
    """
    Generate Gaussian filter responses.
    """
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wavegrid[0], wavegrid[-1], size=n_filters)
    widths = rng.uniform((wavegrid[-1]-wavegrid[0])/20,
                         (wavegrid[-1]-wavegrid[0])/10, size=n_filters)
    filters = np.zeros((n_filters, n_wave))
    for i, (c, w) in enumerate(zip(centers, widths)):
        sigma = w / 2.355  # convert FWHM to sigma
        filters[i] = np.exp(-0.5 * ((wavegrid - c)/sigma)**2)
        filters[i] /= filters[i].sum()  # normalise
    return filters

def compute_photometry(spectra, filters):
    """
    Compute integrated fluxes through each filter for each spectrum.
    """
    # spectra: (n_samples, n_wave)
    # filters: (n_filters, n_wave)
    return spectra @ filters.T  # shape (n_samples, n_filters)

def build_photon_matrix(basis, filters):
    """
    Build matrix mapping basis coefficients to photometric fluxes.
    """
    # basis: (n_wave, n_basis)
    # filters: (n_filters, n_wave)
    return filters @ basis  # shape (n_filters, n_basis)

def reconstruct_spectrum(filters, photometry, basis, alpha=1.0):
    """
    Reconstruct spectra from photometry using Ridge regression.
    """
    A = build_photon_matrix(basis, filters)          # (n_filters, n_basis)
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(A.T, photometry.T)                     # solve for coefficients
    coeffs = ridge.coef_.T                           # (n_samples, n_basis)
    recon = coeffs @ basis.T                        # (n_samples, n_wave)
    return recon, coeffs

def main():
    rng = np.random.default_rng(42)

    # Define wavelength grid
    n_wave = 1000
    wavegrid = np.linspace(400, 800, n_wave)  # nm

    # Step 1: define spectral model
    n_basis = 5
    basis = generate_basis(n_wave, n_basis, rng=rng)

    # Step 2: generate synthetic spectra
    n_samples = 50
    spectra, true_coeffs = generate_synthetic_spectra(basis, n_samples,
                                                     noise_std=0.02,
                                                     rng=rng)

    # Step 3: generate photometric data
    n_filters = 6
    filters = generate_filter_responses(n_filters, n_wave, wavegrid, rng=rng)
    photometry = compute_photometry(spectra, filters)

    # Step 4: reconstruct spectra from photometry
    recon_spectra, recon_coeffs = reconstruct_spectrum(filters,
                                                       photometry,
                                                       basis,
                                                       alpha=0.1)

    # Output quality metrics
    mse = ((spectra - recon_spectra)**2).mean(axis=1)
    print("Mean squared reconstruction error per sample:", mse.mean())
    print("Example true vs. reconstructed coefficient vectors:")
    print(true_coeffs[0])
    print(recon_coeffs[0])

if __name__ == "__main__":
    main()