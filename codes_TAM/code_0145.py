#!/usr/bin/env python3
"""
Minimal spectral reconstruction demo:
    - Define a smooth spectral basis
    - Generate synthetic spectra as linear combinations of basis spectra
    - Produce photometric measurements by integrating over simple filter curves
    - Reconstruct spectra from photometry using linear least‑squares
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------------------
# Wavelength grid
# ----------------------------------------------------------------------
WAVELENGTH_START = 3000   # Å
WAVELENGTH_END   = 10000  # Å
WAVELENGTH_STEP  = 1      # Å
WAVE_GRID = np.arange(WAVELENGTH_START, WAVELENGTH_END + WAVELENGTH_STEP,
                      WAVELENGTH_STEP)          # shape (N_WL,)

DELTA_WL = WAVELENGTH_STEP


# ----------------------------------------------------------------------
# Filter definitions (simple Gaussian approximations)
# ----------------------------------------------------------------------
FILTER_PARAMS = {
    'u': dict(center=3540, width=500),
    'g': dict(center=4770, width=500),
    'r': dict(center=6230, width=500),
    'i': dict(center=7630, width=500),
    'z': dict(center=9130, width=500),
}

def make_filter(wave_grid, center, width):
    """Return a unit‑normalized Gaussian filter transmission curve."""
    return norm.pdf(wave_grid, loc=center, scale=width)

FILTERS = {name: make_filter(WAVE_GRID, **params) for name, params in FILTER_PARAMS.items()}


# ----------------------------------------------------------------------
# Spectral basis
# ----------------------------------------------------------------------
def generate_basis_spectra(n_basis, wave_grid, rng):
    """Create `n_basis` smooth basis spectra as sums of Gaussians."""
    basis = np.zeros((n_basis, len(wave_grid)))
    for i in range(n_basis):
        n_gauss = rng.integers(2, 5)
        spectrum = np.zeros_like(wave_grid, dtype=float)
        for _ in range(n_gauss):
            amp   = rng.uniform(0.5, 1.5)
            center= rng.uniform(WAVELENGTH_START, WAVELENGTH_END)
            width = rng.uniform(50, 400)
            spectrum += amp * norm.pdf(wave_grid, loc=center, scale=width)
        # add weak linear trend
        slope = rng.uniform(-0.0005, 0.0005)
        spectrum += slope * (wave_grid - wave_grid.mean())
        # ensure positivity
        spectrum -= spectrum.min()
        basis[i] = spectrum
    return basis


# ----------------------------------------------------------------------
# Synthetic spectrum generation
# ----------------------------------------------------------------------
def synthesize_spectrum(coeffs, basis, rng):
    """Linear combination of basis spectra + Gaussian noise."""
    spectrum = coeffs @ basis  # shape (len(wave_grid),)
    noise_std = rng.uniform(0.01, 0.03) * spectrum.max()
    noise = rng.normal(scale=noise_std, size=spectrum.shape)
    spectrum += noise
    spectrum[spectrum < 0] = 0  # enforce non‑negative flux
    return spectrum


# ----------------------------------------------------------------------
# Photometry
# ----------------------------------------------------------------------
def photometric_fluxes(spectrum, filters, wave_grid):
    """Compute fluxes in each filter by integrating over the transmission."""
    fluxes = {}
    for name, filt in filters.items():
        num = np.trapz(spectrum * filt, wave_grid)
        den = np.trapz(filt, wave_grid) + 1e-12  # avoid division by zero
        fluxes[name] = num / den
    return np.array([fluxes[name] for name in sorted(filters)])  # order alphabetically


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def construct_filter_matrix(basis, filters, wave_grid):
    """Pre‑compute the integral of each basis spectrum through each filter."""
    n_filters = len(filters)
    n_basis   = basis.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i, (name, filt) in enumerate(sorted(filters.items())):
        for j in range(n_basis):
            num = np.trapz(basis[j] * filt, wave_grid)
            den = np.trapz(filt, wave_grid) + 1e-12
            M[i, j] = num / den
    return M


def reconstruct_spectrum(fluxes, M, basis):
    """Solve for coefficients and build the reconstructed spectrum."""
    coeffs, *_ = np.linalg.lstsq(M, fluxes, rcond=None)
    recon_spec = coeffs @ basis
    recon_spec[recon_spec < 0] = 0
    return recon_spec, coeffs


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # 1. Create basis spectra
    N_BASIS = 10
    basis = generate_basis_spectra(N_BASIS, WAVE_GRID, rng)

    # 2. Pre‑compute filter matrix
    M = construct_filter_matrix(basis, FILTERS, WAVE_GRID)

    # 3. Generate synthetic data
    N_SAMPLES = 5
    all_true_specs = []
    all_recon_specs = []

    for _ in range(N_SAMPLES):
        coeffs_true = rng.uniform(0.0, 1.0, size=N_BASIS)
        true_spec   = synthesize_spectrum(coeffs_true, basis, rng)
        fluxes      = photometric_fluxes(true_spec, FILTERS, WAVE_GRID)

        recon_spec, coeffs_est = reconstruct_spectrum(fluxes, M, basis)

        all_true_specs.append(true_spec)
        all_recon_specs.append(recon_spec)

        # Report per‑sample error
        rmse = np.sqrt(mean_squared_error(true_spec, recon_spec))
        print(f"Sample RMSE: {rmse:.4f}")

    # 4. Overall statistics
    all_true = np.concatenate(all_true_specs)
    all_recon = np.concatenate(all_recon_specs)
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_recon))
    print(f"\nOverall RMSE across all samples: {overall_rmse:.4f}")


if __name__ == "__main__":
    main()