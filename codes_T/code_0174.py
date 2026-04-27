#!/usr/bin/env python
# Minimal reconstruction framework

import numpy as np
from sklearn.linear_model import Ridge

def wavelengths_grid(start=300.0, stop=800.0, num=501):
    """Create a wavelength grid in nm."""
    return np.linspace(start, stop, num)

def gaussian_basis(n_bases, wav, rng=None):
    """Generate Gaussian basis functions."""
    rng = rng or np.random.default_rng()
    bases = []
    for _ in range(n_bases):
        center = rng.uniform(wav[0], wav[-1])
        sigma = rng.uniform((wav[-1] - wav[0]) / (4 * n_bases), (wav[-1] - wav[0]) / (2 * n_bases))
        g = np.exp(-0.5 * ((wav - center) / sigma)**2)
        bases.append(g / np.sqrt(np.trapz(g, wav)))  # normalize
    return np.vstack(bases)  # shape (n_bases, len(wav))

def generate_spectra(n_samples, basis, rng=None):
    """Generate synthetic spectra as linear combinations of basis functions."""
    rng = rng or np.random.default_rng()
    coefs = rng.normal(size=(n_samples, basis.shape[0]))
    spectra = coefs @ basis  # shape (n_samples, len(wav))
    return spectra, coefs

def filter_transmission(filter_spec, wav):
    """Return top‑hat transmission for a filter spec (name, start, end)."""
    name, start, end = filter_spec
    trans = np.zeros_like(wav)
    mask = (wav >= start) & (wav <= end)
    trans[mask] = 1.0
    return trans

def generate_filters(specs, wav):
    """Create list of (name, transmission array)."""
    return [(name, filter_transmission((name, start, end), wav))
            for name, start, end in specs]

def integrate_over_filter(flux, trans, wav):
    """Integrate flux over filter transmission."""
    return np.trapz(flux * trans, wav) / np.trapz(trans, wav)

def generate_photometry(spectra, filters, wav, noise_sigma=0.01, rng=None):
    """Compute photometric fluxes from spectra."""
    rng = rng or np.random.default_rng()
    phot = np.zeros((spectra.shape[0], len(filters)))
    for i, (name, trans) in enumerate(filters):
        for j, spec in enumerate(spectra):
            flux = integrate_over_filter(spec, trans, wav)
            phot[j, i] = flux
    # Add Gaussian noise
    phot += rng.normal(scale=noise_sigma, size=phot.shape)
    return phot

def build_design_matrix(filters, basis, wav):
    """Build matrix mapping coefficients to photometric fluxes."""
    n_filters = len(filters)
    n_bases = basis.shape[0]
    X = np.empty((n_filters, n_bases))
    for i, (name, trans) in enumerate(filters):
        for k in range(n_bases):
            X[i, k] = integrate_over_filter(basis[k], trans, wav)
    return X

def reconstruct_spectra(phot, X, basis, alpha=1.0):
    """Reconstruct spectra from photometry using ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X.T, phot.T)  # X.T shape (n_bases, n_filters)
    recon_coefs = ridge.coef_.T  # shape (n_samples, n_bases)
    recon_spectra = recon_coefs @ basis  # shape (n_samples, len(wav))
    return recon_spectra, recon_coefs

def main():
    rng = np.random.default_rng(42)

    # 1. Define wavelength grid
    wav = wavelengths_grid()

    # 2. Define spectral basis
    basis = gaussian_basis(n_bases=5, wav=wav, rng=rng)

    # 3. Generate synthetic spectra
    n_samples = 20
    spectra, true_coefs = generate_spectra(n_samples, basis, rng=rng)

    # 4. Define filters
    filter_specs = [
        ("U", 300.0, 380.0),
        ("B", 380.0, 500.0),
        ("V", 500.0, 600.0),
        ("R", 600.0, 700.0)
    ]
    filters = generate_filters(filter_specs, wav)

    # 5. Generate photometry
    phot = generate_photometry(spectra, filters, wav, noise_sigma=0.02, rng=rng)

    # 6. Build design matrix and reconstruct spectra
    X = build_design_matrix(filters, basis, wav)
    recon_spectra, recon_coefs = reconstruct_spectra(phot, X, basis, alpha=0.1)

    # 7. Output shapes
    print("Spectra shape:", spectra.shape)
    print("Photometry shape:", phot.shape)
    print("Reconstructed spectra shape:", recon_spectra.shape)

if __name__ == "__main__":
    main()