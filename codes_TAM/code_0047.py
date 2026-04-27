import numpy as np
from sklearn.linear_model import LinearRegression


def make_basis_functions(n_basis, wave_grid):
    """Create polynomial basis functions up to degree n_basis‑1."""
    return [np.power(wave_grid / np.max(wave_grid), n) for n in range(n_basis)]


def generate_synthetic_spectra(n_spectra, basis_funcs, noise_std=0.01, rng=None):
    """Generate synthetic spectra as linear combinations of basis functions."""
    if rng is None:
        rng = np.random.default_rng()
    n_basis = len(basis_funcs)
    coeffs = rng.standard_normal((n_spectra, n_basis)) * 0.1
    basis_matrix = np.vstack(basis_funcs)      # (n_basis, n_wave)
    spectra = coeffs @ basis_matrix           # (n_spectra, n_wave)
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs


def create_gaussian_bandpasses(cen_lams, width, wave_grid):
    """Create Gaussian transmission curves for a set of central wavelengths."""
    bandpasses = []
    for lam in cen_lams:
        trans = np.exp(-0.5 * ((wave_grid - lam) / width) ** 2)
        bandpasses.append(trans)
    return np.array(bandpasses)  # (n_bands, n_wave)


def compute_photometry(spectra, bandpasses, wave_grid):
    """Integrate spectra over bandpasses to obtain photometric fluxes."""
    n_spectra = spectra.shape[0]
    n_bands = bandpasses.shape[0]
    phot = np.empty((n_spectra, n_bands))
    norm = np.trapz(bandpasses, wave_grid, axis=1)   # (n_bands,)
    for i in range(n_bands):
        integ = np.trapz(spectra * bandpasses[i][None, :], wave_grid, axis=1)
        phot[:, i] = integ / norm[i]
    return phot


def reconstruct_spectra(photometry, bandpasses, basis_funcs, wave_grid):
    """Reconstruct spectra from photometry via linear regression on basis functions."""
    n_bands = bandpasses.shape[0]
    n_basis = len(basis_funcs)
    # Design matrix: integration of each basis over each band
    A = np.empty((n_bands, n_basis))
    norm = np.trapz(bandpasses, wave_grid, axis=1)  # (n_bands,)
    for j, bf in enumerate(basis_funcs):
        integ = np.trapz(bandpasses * bf[None, :], wave_grid, axis=1)
        A[:, j] = integ / norm
    # Solve for coefficients for each spectrum
    coeffs, *_ = np.linalg.lstsq(A, photometry.T, rcond=None)  # (n_basis, n_spectra)
    # Reconstruct spectra
    basis_matrix = np.vstack(basis_funcs)  # (n_basis, n_wave)
    recon = coeffs.T @ basis_matrix       # (n_spectra, n_wave)
    return recon, coeffs.T


def main():
    rng = np.random.default_rng(42)
    # Wavelength grid (nm)
    wave = np.linspace(350, 850, 2000)
    # Basis functions
    basis = make_basis_functions(n_basis=5, wave_grid=wave)
    # Synthetic spectra
    n_star = 100
    spectra, true_coeffs = generate_synthetic_spectra(n_star, basis, noise_std=0.02, rng=rng)
    # Bandpasses
    band_centers = np.array([400, 500, 600, 700, 800])   # nm
    band_width = 30.0
    bandpasses = create_gaussian_bandpasses(band_centers, band_width, wave)
    # Photometry
    phot = compute_photometry(spectra, bandpasses, wave)
    # Reconstruction
    recon_spectra, rec_coeffs = reconstruct_spectra(phot, bandpasses, basis, wave)
    # Evaluate
    mae = np.mean(np.abs(recon_spectra - spectra))
    print(f"Mean absolute reconstruction error: {mae:.4f}")


if __name__ == "__main__":
    main()