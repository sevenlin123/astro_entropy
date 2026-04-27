import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Define a spectral model (basis spectra)
# ------------------------------------------------------------------
def generate_basis_spectra(wave, n_basis=5, seed=0):
    """
    Generate a set of simple basis spectra (Gaussian bumps) on the wavelength grid.
    """
    rng = np.random.default_rng(seed)
    basis = []
    for i in range(n_basis):
        center = rng.uniform(wave.min() + 0.1 * (wave.max()-wave.min()),
                             wave.max() - 0.1 * (wave.max()-wave.min()))
        width  = rng.uniform(0.01*(wave.max()-wave.min()), 0.05*(wave.max()-wave.min()))
        amplitude = rng.uniform(0.5, 1.5)
        spec = amplitude * np.exp(-0.5*((wave-center)/width)**2)
        basis.append(spec)
    return np.vstack(basis)  # shape (n_basis, n_wave)

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_random_spectrum(basis, rng=None):
    """
    Create a synthetic spectrum as a random linear combination of basis spectra.
    """
    if rng is None:
        rng = np.random.default_rng()
    coeffs = rng.uniform(0.5, 1.5, size=basis.shape[0])
    spectrum = coeffs @ basis  # shape (n_wave,)
    return spectrum, coeffs

# ------------------------------------------------------------------
# 3. Define simple filter curves
# ------------------------------------------------------------------
def generate_filters(wave, n_filters=6, seed=1):
    """
    Generate Gaussian-shaped filter transmission curves.
    """
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wave.min()+0.1*(wave.max()-wave.min()),
                             wave.max()-0.1*(wave.max()-wave.min()))
        width  = rng.uniform(0.02*(wave.max()-wave.min()), 0.1*(wave.max()-wave.min()))
        trans  = np.exp(-0.5*((wave-center)/width)**2)
        trans /= trans.max()  # normalize to 1
        filters.append(trans)
    return np.vstack(filters)  # shape (n_filters, n_wave)

# ------------------------------------------------------------------
# 4. Compute synthetic photometry
# ------------------------------------------------------------------
def compute_photometry(spectrum, filters):
    """
    Integrate spectrum * filter over wavelength to obtain photometric fluxes.
    """
    # Simple trapezoidal integration
    fluxes = []
    for filt in filters:
        flux = np.trapz(spectrum * filt, x=wave)
        fluxes.append(flux)
    return np.array(fluxes)  # shape (n_filters,)

# ------------------------------------------------------------------
# 5. Reconstruct spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(filters, photometry, basis, alpha=1.0):
    """
    Solve for basis coefficients using ridge regression (least-squares with Tikhonov regularization).
    Then reconstruct the full spectrum as a linear combination of basis spectra.
    """
    # Build design matrix: for each filter, integrate each basis spectrum through it
    n_filters, n_wave = filters.shape
    n_basis = basis.shape[0]
    X = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            X[i, j] = np.trapz(basis[j] * filters[i], x=wave)
    # Solve for coefficients
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, photometry)
    coeffs = reg.coef_
    # Reconstruct spectrum
    reconstructed = coeffs @ basis
    return reconstructed, coeffs

# ------------------------------------------------------------------
# 6. Main routine
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(4000, 7000, 300)  # Angstrom

    # Basis spectra
    basis = generate_basis_spectra(wave, n_basis=5, seed=42)

    # Synthetic photometric filters
    filters = generate_filters(wave, n_filters=6, seed=24)

    # Generate multiple synthetic sources
    n_sources = 10
    all_spectra = []
    all_coeffs  = []
    all_photons = []

    for _ in range(n_sources):
        spec, coeffs = generate_random_spectrum(basis)
        fluxes = compute_photometry(spec, filters)
        all_spectra.append(spec)
        all_coeffs.append(coeffs)
        all_photons.append(fluxes)

    all_spectra   = np.array(all_spectra)   # shape (n_sources, n_wave)
    all_coeffs    = np.array(all_coeffs)    # shape (n_sources, n_basis)
    all_photons   = np.array(all_photons)   # shape (n_sources, n_filters)

    # Reconstruction for each source
    recon_spectra = []
    recon_coeffs  = []
    for fluxes in all_photons:
        recon_spec, recon_coef = reconstruct_spectrum_from_photometry(filters, fluxes, basis)
        recon_spectra.append(recon_spec)
        recon_coeffs.append(recon_coef)

    recon_spectra = np.array(recon_spectra)   # shape (n_sources, n_wave)
    recon_coeffs  = np.array(recon_coeffs)    # shape (n_sources, n_basis)

    # Simple assessment: plot one example
    idx = 0
    plt.figure(figsize=(10, 4))
    plt.plot(wave, all_spectra[idx], label='True Spectrum')
    plt.plot(wave, recon_spectra[idx], '--', label='Reconstructed Spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (arbitrary units)')
    plt.legend()
    plt.title('Spectrum Reconstruction Example')
    plt.tight_layout()
    plt.show()

    # Print comparison of true vs recovered coefficients for the example
    print("True coefficients:", all_coeffs[idx])
    print("Recovered coeffs :", recon_coeffs[idx])