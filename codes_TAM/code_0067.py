import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: Gaussian basis functions
def gaussian_basis(wave, centers, sigma=0.02):
    return np.exp(-0.5 * ((wave - centers[:, None]) / sigma)**2)

# 2. Generate synthetic spectra
def synth_spectrum(n_spectra, wave, n_basis=10, noise=0.01):
    # Random weights for basis functions
    centers = np.linspace(0.4, 0.8, n_basis)  # wavelengths in microns
    B = gaussian_basis(wave, centers)
    coeffs = np.random.randn(n_spectra, n_basis)
    spectra = coeffs @ B
    # Add Gaussian noise
    spectra += noise * np.random.randn(*spectra.shape)
    return spectra, coeffs, B

# 3. Create photometric bands (wavelength ranges) with transmission curves
def photometric_bands(wave, n_bands=5):
    band_centers = np.linspace(0.45, 0.75, n_bands)
    trans = []
    for bc in band_centers:
        trans.append(np.exp(-(wave-bc)**2/(2*0.01**2)))
    return np.array(trans), band_centers

# 4. Generate photometric measurements from synthetic spectra
def compute_photometry(spectra, wave, trans):
    phots = np.array([simps(spectra[:, i] * trans[j], wave) for j in range(len(trans))])
    # For each band we have one measurement per spectrum; shape (n_bands, n_spectra)
    return phots.T

# 5. reconstruct a single synthetic spectrum using ridge regression
def reconstruct_from_photometry(phot, wave, trans, B):
    # Build design matrix: integrate basis over each band
    # For each band j, integral of each basis function over band
    M = np.array([[simps(B[:, i] * trans[j], wave) for i in range(B.shape[1])] for j in range(len(trans))])
    # Solve M * coeffs = phot (for each spectrum)
    ridge = Ridge(alpha=0.1, fit_intercept=False)
    ridge.fit(M, phot)
    coeffs = ridge.coef_.T  # shape (n_basis, n_spectra)
    # Reconstruct spectra from coefficients
    recon = coeffs.T @ B
    return recon

# Main routine
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(0.35, 0.85, 1000)  # in microns
    # Generate synthetic spectra
    spectra, coeffs_true, B = synth_spectrum(3, wave, n_basis=10, noise=0.02)
    # Create photometric bands
    trans, band_centers = photometric_bands(wave, n_bands=5)
    # Compute photometric data
    phot = compute_photometry(spectra, wave, trans)
    # Reconstruct spectra from photometric data
    recon = reconstruct_from_photometry(phot, wave, trans, B)
    # Print results
    print("True coeffs:\n", coeffs_true)
    print("Reconstructed coefficients:\n", recon @ B.T)  # approximate true coeffs
    print("Difference between true and reconstructed spectra:\n", np.linalg.norm(spectra - recon))