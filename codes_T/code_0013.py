#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model utilities
# ------------------------------------------------------------------
def create_wavelength_grid(start=400.0, stop=800.0, num=1000):
    """Create a wavelength grid in nanometers."""
    return np.linspace(start, stop, num)

def gaussian_basis(wav, centers, widths):
    """Generate a list of Gaussian basis functions."""
    return [norm.pdf(wav, loc=c, scale=w) for c, w in zip(centers, widths)]

def normalize_basis(basis, wav):
    """Normalize each basis function to unit integral."""
    return [b / np.trapz(b, wav) for b in basis]

# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------
def generate_true_coeffs(n_basis, rng=np.random.default_rng()):
    """Random coefficients for the true spectrum."""
    return rng.uniform(-1.0, 1.0, size=n_basis)

def synthesize_spectrum(basis, coeffs):
    """Construct a spectrum from basis functions and coefficients."""
    return sum(c * b for c, b in zip(coeffs, basis))

def generate_filters(wav, n_filters=3, rng=np.random.default_rng()):
    """Generate random top‑hat filter transmission curves."""
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wav[0], wav[-1])
        width = rng.uniform((wav[-1]-wav[0]) * 0.05,
                            (wav[-1]-wav[0]) * 0.15)
        filt = np.clip(norm.cdf((wav - center) / width), 0, 1)
        filters.append(filt)
    return filters

def integrate_flux(spectrum, filt, wav):
    """Integrate spectrum weighted by a filter transmission curve."""
    return np.trapz(spectrum * filt, wav)

def generate_photometry(spectrum, filters, wav, snr=30, rng=np.random.default_rng()):
    """Compute noisy photometric fluxes through each filter."""
    noiseless = np.array([integrate_flux(spectrum, f, wav) for f in filters])
    sigma = noiseless / snr
    noise = rng.normal(0, sigma)
    return noiseless + noise

# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------
def build_design_matrix(filters, basis, wav):
    """Build the matrix mapping basis coefficients to photometric fluxes."""
    return np.array([[integrate_flux(b, f, wav) for b in basis]
                     for f in filters])

def reconstruct_spectrum(photometry, design_matrix, basis, wav):
    """Estimate spectrum coefficients from photometry via least‑squares."""
    model = LinearRegression(fit_intercept=False)
    model.fit(design_matrix, photometry)
    coeffs_est = model.coef_
    return synthesize_spectrum(basis, coeffs_est)

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wav = create_wavelength_grid()

    # Basis functions
    centers = np.linspace(450, 750, 5)          # 5 Gaussian components
    widths = np.full_like(centers, 30.0)         # 30 nm width
    basis_raw = gaussian_basis(wav, centers, widths)
    basis = normalize_basis(basis_raw, wav)

    # True coefficients and spectrum
    true_coeffs = generate_true_coeffs(len(basis), rng)
    true_spec = synthesize_spectrum(basis, true_coeffs)

    # Filters
    filters = generate_filters(wav, n_filters=4, rng=rng)

    # Photometric observations
    photometry = generate_photometry(true_spec, filters, wav, snr=20, rng=rng)

    # Design matrix and reconstruction
    design_mat = build_design_matrix(filters, basis, wav)
    recon_spec = reconstruct_spectrum(photometry, design_mat, basis, wav)

    # Print results
    print("True coefficients :", true_coeffs)
    print("Estimated coeffs :", np.round(recon_spec / np.trapz(recon_spec, wav), 3))
    # Simple error metric
    error = np.abs(true_spec - recon_spec).mean()
    print(f"Mean absolute error of reconstructed spectrum: {error:.6f}")

if __name__ == "__main__":
    main()