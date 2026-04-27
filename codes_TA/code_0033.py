import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# 1. Define a spectral model
def generate_spectrum(wavelengths, params):
    """Simple linear combination of Gaussian components."""
    spec = np.zeros_like(wavelengths)
    for amp, cen, wid in params:
        spec += amp * np.exp(-(wavelengths - cen)**2 / (2 * wid**2))
    return spec

# 2. Generate synthetic spectra
def create_synthetic_dataset(n_samples=10, n_bins=200):
    """Generate random spectra with random Gaussian parameters."""
    wavelengths = np.linspace(350, 2500, n_bins)  # nm
    spectra = []
    true_params_list = []
    for _ in range(n_samples):
        n_lines = np.random.randint(1, 6)
        params = [(np.random.uniform(0.5, 1.5),  # amplitude
                   np.random.uniform(400, 2400),  # center
                   np.random.uniform(5, 30))     # width
                  for _ in range(n_lines)]
        spec = generate_spectrum(wavelengths, params)
        spectra.append(spec)
        true_params_list.append(params)
    return wavelengths, np.array(spectra), np.array(true_params_list)

# 3. Photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filters):
    """
    Integrate spectra over filter transmission curves.
    filters: list of (center, width) tuples in nm.
    """
    fluxes = []
    for filt_center, filt_width in filters:
        mask = (wavelengths >= filt_center - filt_width/2) & \
               (wavelengths <= filt_center + filt_width/2)
        flux = np.trapz(spectra[:, mask], wavelengths[mask], axis=1)
        fluxes.append(flux)
    return np.vstack(fluxes).T

# 4. Spectrum reconstruction
def reconstruct_spectrum(photon_counts, wavelengths, filters, n_components=3):
    """Reconstruct spectrum using Lasso regression on basis functions."""
    # Build design matrix A (filters x wavelengths)
    N = len(filters)
    M = len(wavelengths)
    A = np.zeros((N, M))
    for i, (cen, wid) in enumerate(filters):
        A[i] = np.exp(-(wavelengths - cen)**2 / (2 * wid**2))
    # Lasso regression
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(A, photon_counts)
    # Reconstructed spectrum (weighted sum of basis Gaussians)
    reconstructed = lasso.coef_
    return reconstructed

# Main routine
if __name__ == "__main__":
    # Create synthetic dataset
    wav, specs, true_params = create_synthetic_dataset(n_samples=5, n_bins=300)

    # Define simple filter set (centers & widths)
    filter_set = [(450, 50), (550, 60), (650, 70), (750, 80)]

    # Compute photometric observations
    phot = compute_photometry(specs, wav, filter_set)

    # Reconstruct one spectrum (choose first)
    idx = 0
    recon = reconstruct_spectrum(phot[idx], wav, filter_set, n_components=3)

    # Compare to original
    import matplotlib.pyplot as plt
    plt.plot(wav, specs[idx], label='original')
    plt.plot(wav, recon, '--', label='reconstructed')
    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    plt.title('Spectrum Reconstruction Example')
    plt.show()