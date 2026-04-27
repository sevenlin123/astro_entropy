import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------------------------
# 1. Spectral model
# ---------------------------

def gaussian_spectrum(wave, amp, cen, sigma):
    """Single Gaussian component."""
    return amp * np.exp(-0.5 * ((wave - cen) / sigma)**2)

def composite_spectrum(wave, params):
    """
    params : list of tuples (amp, cen, sigma)
    Returns a sum of Gaussian components.
    """
    spec = np.zeros_like(wave)
    for amp, cen, sigma in params:
        spec += gaussian_spectrum(wave, amp, cen, sigma)
    return spec

# ---------------------------
# 2. Generate synthetic spectra
# ---------------------------

def generate_basis_spectra(n_basis, wave, rng=None):
    """Generate n_basis synthetic spectra with random Gaussian components."""
    if rng is None:
        rng = np.random.default_rng()
    basis = []
    for _ in range(n_basis):
        # Randomly choose 1-3 components per spectrum
        n_comp = rng.integers(1, 4)
        params = []
        for _ in range(n_comp):
            amp = rng.uniform(0.5, 1.5)
            cen = rng.uniform(wave.min(), wave.max())
            sigma = rng.uniform(5, 20)
            params.append((amp, cen, sigma))
        basis.append(composite_spectrum(wave, params))
    return np.array(basis)  # shape (n_basis, n_wave)

# ---------------------------
# 3. Generate photometric data
# ---------------------------

def gaussian_filter(wave, cen, sigma):
    """Filter transmission profile: Gaussian."""
    return np.exp(-0.5 * ((wave - cen) / sigma)**2)

def photometry_from_spectrum(spectrum, filters, wave):
    """
    spectrum : array of fluxes
    filters : list of (cen, sigma) tuples
    Returns integrated flux in each filter.
    """
    ph = []
    for cen, sigma in filters:
        filt = gaussian_filter(wave, cen, sigma)
        ph.append(np.trapz(spectrum * filt, wave) / np.trapz(filt, wave))
    return np.array(ph)

# ---------------------------
# 4. Reconstruction framework
# ---------------------------

def build_photometry_matrix(basis_spectra, filters, wave):
    """Compute photometry for all basis spectra."""
    n_basis = basis_spectra.shape[0]
    phot_mat = np.zeros((len(filters), n_basis))
    for i, spec in enumerate(basis_spectra):
        phot_mat[:, i] = photometry_from_spectrum(spec, filters, wave)
    return phot_mat  # shape (n_filters, n_basis)

def reconstruct_spectrum(target_phot, phot_mat, basis_spectra, alpha=1e-3):
    """
    Reconstruct spectrum as linear combo of basis_spectra.
    Uses Ridge regression to find coefficients.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(phot_mat.T, target_phot)
    coeffs = ridge.coef_
    recon = coeffs @ basis_spectra
    return recon, coeffs

# ---------------------------
# 5. Demo
# ---------------------------

def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    wave = np.linspace(400, 700, 300)   # 400-700 nm

    # Filters (central wavelength, width)
    filters = [(450, 20), (550, 20), (650, 20)]

    # Generate basis spectra
    n_basis = 5
    basis_spectra = generate_basis_spectra(n_basis, wave, rng)

    # Build photometry matrix for basis
    phot_mat = build_photometry_matrix(basis_spectra, filters, wave)

    # Create a target spectrum as random linear combo of basis
    true_coeffs = rng.normal(size=n_basis)
    target_spec = true_coeffs @ basis_spectra

    # Obtain photometric measurements for target
    target_phot = photometry_from_spectrum(target_spec, filters, wave)

    # Reconstruct spectrum
    recon_spec, recon_coeffs = reconstruct_spectrum(target_phot, phot_mat, basis_spectra)

    # Output results
    print("True coefficients:     ", true_coeffs)
    print("Reconstructed coeffs: ", recon_coeffs)
    print("\nPhotometry comparison:")
    print("Target photometry :", target_phot)
    print("Reconstructed phot", photometry_from_spectrum(recon_spec, filters, wave))

if __name__ == "__main__":
    main()