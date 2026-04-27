import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# 1. Define a simple spectral model: linear combination of basis spectra
def define_spectral_basis(n_basis=3, n_wave=1000):
    """Create synthetic basis spectra over wavelength grid."""
    wl = np.linspace(4000, 8000, n_wave)  # Angstroms
    basis = []
    for i in range(n_basis):
        center = 4000 + i * 2000 / (n_basis - 1)
        sigma = 200
        gauss = np.exp(-0.5 * ((wl - center) / sigma)**2)
        basis.append(gauss)
    return wl, np.array(basis)

# 2. Generate synthetic spectra (coefficients drawn from normal distribution)
def generate_synthetic_spectra(basis, n_objects=50):
    """Generate synthetic spectra as linear combinations of basis spectra."""
    n_basis, n_wave = basis.shape
    coeffs = np.random.randn(n_objects, n_basis)
    spectra = coeffs @ basis  # (n_objects, n_wave)
    return coeffs, spectra

# 3. Generate photometric fluxes from spectra using simple bandpasses
def generate_photometry(spectra, wl, band_edges=[(5000, 5500), (6000, 7000)]):
    """Compute photometric fluxes by integrating spectra over bandpasses."""
    fluxes = []
    for (lo, hi) in band_edges:
        mask = (wl >= lo) & (wl <= hi)
        band_flux = spectra[:, mask].mean(axis=1)  # simple mean over band
        fluxes.append(band_flux)
    return np.column_stack(fluxes)

# 4. reconstruct spectra from photometry using Ridge regression
def reconstruct_from_photometry(pheno, basis):
    """Ridge regression to find coefficients that best reproduce photometry."""
    # build response matrix from basis projected into bands
    # Assume we know band edges used earlier
    wl = np.linspace(4000, 8000, basis.shape[1])
    band_edges = [(5000, 5500), (6000, 7000)]
    A = []
    for (lo, hi) in band_edges:
        mask = (wl >= lo) & (wl <= hi)
        band_spectrum = basis[:, mask].mean(axis=1)  # average spectrum per band
        A.append(band_spectrum)
    A = np.vstack(A).T  # shape (n_objects? no, shape (n_samples, n_basis))
    # RidgeCV with alpha selection
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    ridge.fit(A.T, pheno.T)  # fit each photometric band separately? 
    # We want coefficient matrix: n_objects x n_basis
    coeffs_est = ridge.coef_.T
    # reconstruct spectra
    spectra_recon = coeffs_est @ basis
    return spectra_recon

# Main execution
if __name__ == "__main__":
    wl, basis = define_spectral_basis()
    coeffs_true, spectra_true = generate_synthetic_spectra(basis)
    photometry = generate_photometry(spectra_true, wl)
    recon_spectra = reconstruct_from_photometry(photometry, basis)
    # simple check: print RMS error between true and reconstructed spectra
    rms_error = np.sqrt(((spectra_true - recon_spectra)**2).mean())
    print(f"Reconstruction RMS error: {rms_error:.4f}")